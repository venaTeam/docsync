"""Shared data contracts for the docsync pipeline.

Every stage of the pipeline (diff -> impact -> edits -> validate -> PR) exchanges
the dataclasses / pydantic models defined here. This module is the single seam
all other modules build against; keep it dependency-light (pydantic + stdlib).

Pydantic v2 is used (this is a standalone tool, unrelated to Keep's pydantic v1
services) because the Anthropic SDK's `messages.parse()` structured-output helper
validates against pydantic models directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Stage 2 — diff extraction (diff.py)
# ---------------------------------------------------------------------------


class FileStatus(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"


class ChangedFile(BaseModel):
    """One file touched by a diff, with its hunks and the code symbols affected."""

    path: str
    status: FileStatus
    previous_path: Optional[str] = None  # set when status == RENAMED
    # Unified-diff hunk texts (the `@@ ... @@` blocks + context), one per hunk.
    hunks: list[str] = Field(default_factory=list)
    # Function / class / module-level assignment names whose body the hunks touch.
    # The cross-boundary signal used by impact mapping; survives line-number churn.
    changed_symbols: list[str] = Field(default_factory=list)


class CodeDiff(BaseModel):
    """The structured result of comparing base..head in a single service repo."""

    repo: str  # e.g. "keephq/keep-api-gateway" (owner/name) or a local path
    base_sha: str
    head_sha: str
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    files: list[ChangedFile] = Field(default_factory=list)

    def changed_paths(self) -> list[str]:
        out: list[str] = []
        for f in self.files:
            out.append(f.path)
            if f.previous_path:
                out.append(f.previous_path)
        return out

    def all_symbols(self) -> list[str]:
        seen: list[str] = []
        for f in self.files:
            for s in f.changed_symbols:
                if s not in seen:
                    seen.append(s)
        return seen


# ---------------------------------------------------------------------------
# Bootstrap — whole-repo ingest + doc planning (ingest.py / bootstrap.py)
# ---------------------------------------------------------------------------


class SourceUnit(BaseModel):
    """One documentable source file, distilled to what the planner needs.

    Lightweight by design: paths + symbol names only (no file bodies). A whole
    service repo's worth of these has to fit in the planner's context, and the
    excerpts are fetched per-page only at author time.
    """

    path: str  # repo-relative path, e.g. "src/routes/alerts.py"
    kind: str  # coarse language/role tag: "python" | "typescript" | "other"
    symbols: list[str] = Field(default_factory=list)  # top-level defs/classes/exports


class RepoDigest(BaseModel):
    """The lightweight, whole-repo snapshot bootstrap plans from."""

    repo: str  # owner/name or local path (mirrors CodeDiff.repo)
    root: str  # absolute path the units were walked from
    units: list[SourceUnit] = Field(default_factory=list)

    def all_symbols(self) -> list[str]:
        seen: list[str] = []
        for u in self.units:
            for s in u.symbols:
                if s not in seen:
                    seen.append(s)
        return seen


# Page kinds steer both the author prompt and how the page is kept live:
#   reference — code-anchored API/data-model pages (precise anchors, judge-autopass)
#   concept   — narrative explanations of a subsystem/architecture (broad anchors, judged)
#   guide     — task-oriented (getting-started/how-to), loosely anchored, judged
PageKind = Literal["concept", "guide", "reference"]


class PlannedSource(BaseModel):
    """A code anchor for a planned page — repo-qualified, for multi-repo sites."""

    repo: str  # which source repo (matches a RepoDigest.repo / CodeDiff.repo)
    globs: list[str] = Field(default_factory=list)  # fnmatch globs over changed paths
    symbols: list[str] = Field(default_factory=list)  # symbol names (trailing * = prefix)


class PlannedPage(BaseModel):
    """One doc page the planner proposes to author (structured LLM output)."""

    page_path: str  # new .mdx path relative to docs_root, e.g. "reference/alerts.mdx"
    title: str
    kind: PageKind = "reference"
    section: str = "Reference"  # nav group / section heading
    order: int = 0  # position within the section (ascending)
    summary: str = ""  # what the page should cover (steers the author stage)
    sources: list[PlannedSource] = Field(default_factory=list)  # multi-repo anchors

    @property
    def judge_required(self) -> bool:
        """Narrative pages route through the judge (not anchor-autopass) when updated.

        A concept/guide page anchors to a whole subsystem, so an autopass would fire a
        costly Opus edit on every change there; routing through the judge means an edit
        only happens when the change actually invalidates the page.
        """
        return self.kind in ("concept", "guide")


# Canonical section order for a generated site's reading flow; unknown sections sort
# after these (stable by first appearance). Used to emit nav groups in sequence.
SECTION_ORDER: tuple[str, ...] = (
    "Getting Started",
    "Concepts",
    "Architecture",
    "Reference",
    "Operations",
)


class DocPlan(BaseModel):
    """The planner's structured response: a flat, section-tagged set of pages.

    Kept flat (not nested sections) because flat structured output is far more reliable
    through the CLI backend; ordered sections are derived in code via `ordered_sections`.
    """

    pages: list[PlannedPage] = Field(default_factory=list)

    def ordered_sections(self) -> list[tuple[str, list[PlannedPage]]]:
        """Group pages into sections, ordered by the canonical reading flow.

        Sections in `SECTION_ORDER` come first in that order; any others follow in
        first-appearance order. Pages within a section are sorted by `order` then title.
        """
        seen: list[str] = []
        buckets: dict[str, list[PlannedPage]] = {}
        for p in self.pages:
            if p.section not in buckets:
                buckets[p.section] = []
                seen.append(p.section)
            buckets[p.section].append(p)

        def section_key(name: str) -> tuple[int, int]:
            canonical = SECTION_ORDER.index(name) if name in SECTION_ORDER else len(SECTION_ORDER)
            return (canonical, seen.index(name))

        out: list[tuple[str, list[PlannedPage]]] = []
        for name in sorted(seen, key=section_key):
            pages = sorted(buckets[name], key=lambda p: (p.order, p.title))
            out.append((name, pages))
        return out


class AuthoredPage(BaseModel):
    """The author stage's structured response: one full MDX page body."""

    content: str  # the complete .mdx file text (frontmatter + body)


# ---------------------------------------------------------------------------
# Config & manifest (config.py) — the page <-> source mapping
# ---------------------------------------------------------------------------


class ManifestSource(BaseModel):
    """One source-of-truth location a doc page is anchored to."""

    repo: str  # matches CodeDiff.repo
    globs: list[str] = Field(default_factory=list)  # fnmatch globs over changed paths
    symbols: list[str] = Field(default_factory=list)  # symbol names (supports trailing *)


class ManifestPage(BaseModel):
    """A documentation page and the code it is anchored to."""

    path: str  # path to the .mdx, relative to the docs repo root
    sources: list[ManifestSource] = Field(default_factory=list)
    max_diff_lines: int = 60  # diff-size guardrail (net changed lines per page)
    max_diff_pct: float = 0.5  # ...or this fraction of the page, whichever is larger
    allow_frontmatter_edit: bool = False
    # When True, an anchor hit does NOT autopass — the page is routed through the judge
    # so an edit fires only on a confirmed invalidation. Set for narrative (concept/
    # guide) pages whose broad subsystem anchors would otherwise over-trigger.
    judge_required: bool = False


class Manifest(BaseModel):
    pages: list[ManifestPage] = Field(default_factory=list)

    def page(self, path: str) -> Optional[ManifestPage]:
        return next((p for p in self.pages if p.path == path), None)


class ModelConfig(BaseModel):
    edit_model: str = "claude-opus-4-8"
    judge_model: str = "claude-haiku-4-5"
    edit_effort: str = "high"


class DocsyncConfig(BaseModel):
    """Loaded from <docs-repo>/.docsync/config.yml (all fields optional)."""

    models: ModelConfig = Field(default_factory=ModelConfig)
    docs_root: str = "."  # root of the docs tree, relative to the docs repo
    # Extra directory names to prune during ingest, on top of ingest.DEFAULT_EXCLUDE_DIRS.
    # Use this to skip repo-specific noise that isn't the product surface you document
    # (e.g. "examples", "deploy", "docs", a generated site dir) so it doesn't inflate
    # the plan or token cost. Matched by directory *name* anywhere in the tree.
    ingest_exclude_dirs: list[str] = Field(default_factory=list)
    # When the judge's confidence is >= this, an embedding/judge candidate is kept.
    judge_confidence_threshold: float = 0.5
    # Anchor hits at or above this judge confidence skip the judge entirely.
    anchor_autopass: bool = True
    reviewers: list[str] = Field(default_factory=list)
    # --- ship-safety dial ---
    # Skip the (expensive) edit stage for any page whose impact confidence is below
    # this. 0.0 = off (anchor autopass is always 1.0, so this only gates judge/
    # embedding pages). Raise it for a conservative first rollout on a real repo.
    min_edit_confidence: float = 0.0
    # Labels applied to opened docs PRs (auto-created in the docs repo if missing).
    pr_labels: list[str] = Field(default_factory=lambda: ["docsync"])
    # Max concurrent LLM requests for the judge + edit stages (a real PR touches
    # several pages; they're independent). Kept low to respect fresh-org rate limits.
    max_parallel_requests: int = 4
    # Hard cap on pages sent to the (expensive) edit stage per run; 0 = unlimited.
    # Highest-confidence pages are edited first; the rest are reported, not edited.
    max_pages_per_run: int = 0
    # Identifier tokens that name source concepts but are too generic to embed well;
    # excluded from the embedding query (e.g. "self", "config", "value").
    stopword_symbols: list[str] = Field(default_factory=list)
    # --- embeddings recall-net (optional; needs the `embeddings` extra) ---
    # sentence-transformers model id used to embed doc chunks + the diff query.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Minimum cosine similarity for an embedding candidate to be worth a judge call.
    embedding_floor: float = 0.2
    # Max embedding candidates surfaced per diff (before the judge filters them).
    embedding_top_k: int = 5


# ---------------------------------------------------------------------------
# Stage 3 — doc-impact mapping (impact.py)
# ---------------------------------------------------------------------------


class CandidateSource(str, Enum):
    ANCHOR = "anchor"
    EMBEDDING = "embedding"


class ImpactCandidate(BaseModel):
    """A page that *might* be affected, before the judge confirms it."""

    page_path: str
    source: CandidateSource
    score: float = 0.0  # anchor: match strength; embedding: cosine similarity
    reason: str = ""  # which glob/symbol matched, or top embedding terms


class JudgeVerdict(BaseModel):
    """Structured output from the Haiku relevance judge (one per candidate page)."""

    page_path: str
    affected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ImpactedPage(BaseModel):
    """A page confirmed (by anchor autopass or judge) to need an update."""

    page_path: str
    source: CandidateSource
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Stage 4 — LLM edit generation (edits.py)
# ---------------------------------------------------------------------------


class EditOp(BaseModel):
    """A single surgical str-replace edit on a page.

    `find` must match the current page text exactly and uniquely; the applier
    rejects not-found or ambiguous ops rather than fuzzy-matching.
    """

    find: str
    replace: str
    rationale: str


class PageEdit(BaseModel):
    """The model's structured response for one page: edits, or a no-change reason."""

    edits: list[EditOp] = Field(default_factory=list)
    no_change_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 5 — validation (validate.py)
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    page_path: str
    passed: bool
    failures: list[str] = Field(default_factory=list)  # hard-gate failures (drop page)
    warnings: list[str] = Field(default_factory=list)  # soft gates (annotate PR)


# ---------------------------------------------------------------------------
# Cost / usage accounting (cost.py)
# ---------------------------------------------------------------------------


class ModelUsage(BaseModel):
    """Accumulated token usage + estimated cost for one (model, stage) across a run."""

    model: str
    stage: Optional[str] = None  # "judge" | "edit" | "critique" | None (unattributed)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


class RunUsage(BaseModel):
    """Token + estimated-cost totals for one run, broken down per model.

    `cost_usd` is computed from a built-in price table (see cost.py) and can drift
    from the real bill — `estimated` is always True to flag that.
    """

    by_model: list[ModelUsage] = Field(default_factory=list)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit_rate: float = 0.0  # cache_read / total prompt tokens, across the run
    estimated: bool = True

    @property
    def prompt_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


# ---------------------------------------------------------------------------
# Pipeline outcome (pipeline.py)
# ---------------------------------------------------------------------------


class PageOutcome(BaseModel):
    """End-to-end result for one impacted page."""

    page_path: str
    impacted: Optional[ImpactedPage] = None  # None for bootstrap-authored pages
    edit: Optional[PageEdit] = None
    validation: Optional[ValidationResult] = None
    new_content: Optional[str] = None  # patched file text, if edits applied & valid
    applied: bool = False  # edits applied and all hard gates passed
    note: str = ""  # human-readable status (dropped reason, no-change reason, etc.)


class PipelineResult(BaseModel):
    diff: CodeDiff
    outcomes: list[PageOutcome] = Field(default_factory=list)
    usage: Optional[RunUsage] = None  # token/cost accounting for the run's LLM calls

    def changed(self) -> list[PageOutcome]:
        return [o for o in self.outcomes if o.applied and o.new_content is not None]


class BootstrapResult(BaseModel):
    """End-to-end result of `docsync bootstrap` over one source repo."""

    repo: str
    plan: DocPlan = Field(default_factory=DocPlan)  # the (deduped, capped) plan
    skipped: list[str] = Field(default_factory=list)  # planned paths dropped pre-author
    outcomes: list[PageOutcome] = Field(default_factory=list)  # one per authored page
    usage: Optional[RunUsage] = None

    def authored(self) -> list[PageOutcome]:
        """Pages that were authored, validated, and are ready to write."""
        return [o for o in self.outcomes if o.applied and o.new_content is not None]
