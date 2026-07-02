"""Shared data contracts for the docsync pipeline.

Every stage of the pipeline (diff -> impact -> edits -> validate -> PR) exchanges
the dataclasses / pydantic models defined here. This module is the single seam
all other modules build against; keep it dependency-light (pydantic + stdlib).

Pydantic v2 is used (this is a standalone tool, unrelated to Keep's pydantic v1
services) because the Anthropic SDK's `messages.parse()` structured-output helper
validates against pydantic models directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """First-seen-wins dedupe (``dict.fromkeys`` keeps insertion order)."""
    return list(dict.fromkeys(items))


def _prompt_tokens(usage) -> int:
    """Billable prompt tokens = fresh input + cache writes + cache reads."""
    return (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )


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
        return _dedupe_preserving_order(s for f in self.files for s in f.changed_symbols)


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
        return _dedupe_preserving_order(s for u in self.units for s in u.symbols)


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

    repo: str = Field(
        default="",
        description=(
            "Source repo this anchor belongs to (matches CodeDiff.repo). Empty = the "
            "only/local repo: matches any diff, so mono/single manifests can omit it. "
            "Poly manifests set it explicitly to scope each anchor to one repo."
        ),
    )
    globs: list[str] = Field(
        default_factory=list, description="fnmatch globs over changed file paths."
    )
    symbols: list[str] = Field(
        default_factory=list, description="Symbol names to match (trailing * = prefix)."
    )


class ManifestPage(BaseModel):
    """A documentation page and the code it is anchored to."""

    path: str = Field(description="Path to the .mdx, relative to the docs repo root.")
    sources: list[ManifestSource] = Field(
        default_factory=list, description="Source anchors that keep this page live."
    )
    max_diff_lines: int = Field(
        default=60, description="Diff-size guardrail: net changed lines allowed per page."
    )
    max_diff_pct: float = Field(
        default=0.5, description="...or this fraction of the page, whichever is larger."
    )
    allow_frontmatter_edit: bool = Field(
        default=False, description="Allow edits to the page's frontmatter title/description."
    )
    judge_required: bool = Field(
        default=False,
        description=(
            "When True, an anchor hit does NOT autopass — the page is routed through the "
            "judge so an edit fires only on a confirmed invalidation. Set for narrative "
            "(concept/guide) pages whose broad anchors would otherwise over-trigger."
        ),
    )


class Manifest(BaseModel):
    pages: list[ManifestPage] = Field(default_factory=list)

    def page(self, path: str) -> Optional[ManifestPage]:
        return next((p for p in self.pages if p.path == path), None)


class ModelConfig(BaseModel):
    edit_model: str = Field(
        default="claude-opus-4-8", description="Model for authoring + surgical edits (Opus)."
    )
    judge_model: str = Field(
        default="claude-haiku-4-5",
        description="Model for the relevance judge, self-critique, and infer (Haiku).",
    )
    edit_effort: str = Field(
        default="high", description="Opus reasoning effort for author + edit calls."
    )


class DocsyncConfig(BaseModel):
    """Loaded from <docs-repo>/.docsync/config.yml (all fields optional).

    Unknown keys are rejected (a typo'd field is an error, not silently ignored). Run
    `docsync explain` to see every field, its type, default, and meaning.
    """

    model_config = ConfigDict(extra="forbid")

    models: ModelConfig = Field(
        default_factory=ModelConfig, description="LLM model choices (see ModelConfig)."
    )
    docs_root: str = Field(
        default=".", description="Root of the docs tree, relative to the docs repo."
    )
    repo_mode: Literal["auto", "mono", "single", "poly"] = Field(
        default="auto",
        description=(
            "Repository topology. mono = docs+code in one checkout (the docs subtree is "
            "filtered out of the diff). single = one code repo + a separate docs repo. "
            "poly = many code repos + one docs repo (run once per repo). auto = detect "
            "from the checkout + manifest."
        ),
    )
    adapter: str = Field(
        default="mintlify",
        description=(
            "Doc-framework adapter that owns the pages: 'mintlify' (.mdx + docs.json nav), "
            "'docusaurus' (.md/.mdx admonitions + sidebars.js nav, air-gapped-friendly), "
            "or 'markdown' (plain .md). Drives frontmatter freeze, structural integrity, "
            "link checks, and the new-page extension."
        ),
    )
    forge: Literal["auto", "github", "gitlab"] = Field(
        default="auto",
        description=(
            "Host for the docs review request opened by --open-pr: 'github' (PR via the "
            "`gh` CLI) or 'gitlab' (MR via the `glab` CLI). 'auto' detects it from the docs "
            "repo's origin remote (URLs containing 'gitlab' → gitlab, else github). Set it "
            "explicitly for a self-managed GitLab on an opaque hostname."
        ),
    )
    thoroughness: Literal["light", "medium", "high"] = Field(
        default="medium",
        description=(
            "Generation thoroughness — how much content to write. light = only the most "
            "important surface (short pages, tighter edits); medium = balanced; high = "
            "exhaustive (every symbol/option/edge case, looser edits)."
        ),
    )
    thoroughness_by_kind: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-page-kind overrides for `thoroughness`, keyed by PageKind (reference | "
            "concept | guide). Applies where a kind is known (bootstrap authoring)."
        ),
    )
    ingest_exclude_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "Extra directory names to prune during ingest (on top of the built-in set), "
            "matched by name anywhere in the tree. Use to skip non-product noise."
        ),
    )
    judge_confidence_threshold: float = Field(
        default=0.5,
        description="Min judge confidence to keep an embedding/judge candidate.",
    )
    anchor_autopass: bool = Field(
        default=True, description="Anchor hits skip the judge entirely."
    )
    reviewers: list[str] = Field(
        default_factory=list, description="GitHub handles requested as reviewers on docs PRs."
    )
    min_edit_confidence: float = Field(
        default=0.0,
        description=(
            "Ship-safety dial: skip the edit stage for pages below this impact confidence. "
            "0 = off. Raise it for a conservative first rollout on a real repo."
        ),
    )
    pr_labels: list[str] = Field(
        default_factory=lambda: ["docsync"],
        description="Labels applied to opened docs PRs (auto-created in the docs repo).",
    )
    max_parallel_requests: int = Field(
        default=4, description="Max concurrent LLM requests across the judge + edit stages."
    )
    max_pages_per_run: int = Field(
        default=0,
        description=(
            "Hard cap on pages sent to the edit stage per run; 0 = unlimited. "
            "Highest-confidence pages are edited first; the rest are reported, not edited."
        ),
    )
    readability_pass: bool = Field(
        default=False,
        description=(
            "Opt-in fact-frozen readability pass after each authored/edited page (one extra "
            "edit-model call per page). The CLI --polish flag toggles it."
        ),
    )
    self_critique: bool = Field(
        default=True,
        description=(
            "Adversarial pass that drops edit ops not justified by the diff (one cheap judge "
            "call per edited page). ON by default; the CLI --no-self-critique disables it."
        ),
    )
    stopword_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Generic identifier tokens excluded from the embedding query (e.g. 'self', "
            "'config', 'value')."
        ),
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="sentence-transformers model id for the embeddings recall-net.",
    )
    embedding_floor: float = Field(
        default=0.2,
        description="Min cosine similarity for an embedding candidate to warrant a judge call.",
    )
    embedding_top_k: int = Field(
        default=5, description="Max embedding candidates surfaced per diff before judging."
    )
    monthly_budget_usd: Optional[float] = Field(
        default=None,
        description=(
            "Soft monthly spend target (USD) shown on the dashboard. Advisory only — never "
            "blocks a run. None = no budget tracking."
        ),
    )

    def thoroughness_for(self, kind: str | None = None) -> str:
        """Effective thoroughness level for a page *kind* (falls back to the global level).

        Pass a PageKind to honor a `thoroughness_by_kind` override; pass nothing (the run
        edit flow, which has no per-page kind) to get the global `thoroughness`.
        """
        if kind:
            return self.thoroughness_by_kind.get(kind, self.thoroughness)
        return self.thoroughness


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
        return _prompt_tokens(self)


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
        return _prompt_tokens(self)


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


# ---------------------------------------------------------------------------
# Run history (history.py) — one distilled record per run, for the dashboard
# ---------------------------------------------------------------------------


DashboardCommand = Literal["run", "bootstrap"]
# "opened" = a PR was opened; "patched" = changes written without a PR; "no_change" =
# a clean run that produced nothing to write; "dry_run" = computed only.
RunStatus = Literal["opened", "patched", "no_change", "dry_run", "error"]


class PageRecord(BaseModel):
    """A distilled, history-safe summary of one page's outcome.

    Deliberately omits page/source body text: only counts and the model's per-edit
    `rationale` are kept, so the persisted history never carries doc content (which
    could contain secrets) and stays one tiny line per run.
    """

    page_path: str
    applied: bool = False
    note: str = ""
    edit_count: int = 0
    rationales: list[str] = Field(default_factory=list)  # EditOp.rationale — the "why"
    # A bounded unified diff of the page (original -> new) — the "what", so the dashboard
    # can show the actual change, not just a summary. Capped in history.py; None when no
    # before-text is available (e.g. bootstrap-authored pages). Doc content only (the
    # published artifact) — never raw source excerpts or the full untouched page body.
    diff: Optional[str] = None
    validation_passed: Optional[bool] = None
    warnings: list[str] = Field(default_factory=list)


class ModelStageRecord(BaseModel):
    """Per-(model, stage) cost summary for one run — the dashboard sums these."""

    model: str
    stage: Optional[str] = None
    calls: int = 0
    tokens: int = 0  # billable prompt tokens + output tokens
    cost_usd: float = 0.0


class UsageRecord(BaseModel):
    """A flattened, dashboard-friendly view of a run's `RunUsage`."""

    cost_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_rate: float = 0.0
    estimated: bool = True
    by_model: list[ModelStageRecord] = Field(default_factory=list)


class RunRecord(BaseModel):
    """One persisted docsync run — the unit the dashboard aggregates.

    `counts` is a flat dict so `run` and `bootstrap` share one shape: a `run` fills
    impacted/updated/dropped; a `bootstrap` fills planned/authored/skipped.
    """

    schema_version: int = 1
    timestamp: str  # ISO-8601 UTC
    command: DashboardCommand
    status: RunStatus = "opened"
    repo: str
    base_sha: Optional[str] = None
    head_sha: Optional[str] = None
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    pr_url: Optional[str] = None
    counts: dict[str, int] = Field(default_factory=dict)
    pages: list[PageRecord] = Field(default_factory=list)
    usage: Optional[UsageRecord] = None


# ---------------------------------------------------------------------------
# Manifest inference (infer.py) — the brownfield analogue of bootstrap
# ---------------------------------------------------------------------------


class InferredSource(BaseModel):
    """One proposed code anchor for an existing page (the judge's raw proposal)."""

    repo: str  # must match one of the ingested RepoDigest.repo ids
    globs: list[str] = Field(default_factory=list)  # fnmatch globs over repo-relative paths
    symbols: list[str] = Field(default_factory=list)  # symbol names (trailing * = prefix)


class InferredAnchors(BaseModel):
    """The judge's structured output: anchors for one existing doc page.

    `kind` drives `judge_required` downstream (concept/guide route through the judge so
    their broad anchors don't over-trigger Opus edits). `confidence` gates whether the
    anchors are written: below `judge_confidence_threshold` the page is reported, not
    anchored.
    """

    page_path: str
    # `confidence` is required (no default) on purpose: it keeps the model from having a
    # single required str field, which the claude-code backend would mis-route as a
    # whole-document output (stuffing the raw JSON into `page_path`). The judge is asked
    # for a calibrated confidence on every page regardless.
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[InferredSource] = Field(default_factory=list)
    kind: PageKind = "reference"
    reason: str = ""


InferStatus = Literal["anchored", "low_confidence", "no_match"]


class InferredPage(BaseModel):
    """Post-validation result for one page (internal — not an LLM output).

    `sources` here are sanitized `ManifestSource`s: every glob matched >=1 real file and
    every symbol appears in the matched code, so an `anchored` page passes `doctor` as-is.
    """

    page_path: str
    sources: list[ManifestSource] = Field(default_factory=list)
    kind: PageKind = "reference"
    confidence: float = 0.0
    status: InferStatus = "no_match"
    reason: str = ""

    @property
    def judge_required(self) -> bool:
        """Narrative pages (concept/guide) route through the judge when updated."""
        return self.kind in ("concept", "guide")


class InferResult(BaseModel):
    """End-to-end result of `docsync infer` over an existing docs site."""

    pages: list[InferredPage] = Field(default_factory=list)
    skipped_already_anchored: list[str] = Field(default_factory=list)
    usage: Optional[RunUsage] = None

    def anchored(self) -> list[InferredPage]:
        """Pages with validated anchors, ready to merge into the manifest."""
        return [p for p in self.pages if p.status == "anchored"]
