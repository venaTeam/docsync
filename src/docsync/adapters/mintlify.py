"""Mintlify doc-framework adapter.

Mintlify pages are `.mdx` (also `.md`) files with YAML frontmatter and inline
MDX components (`<Card>`, `<CardGroup>`, `<Steps>`, ...). This adapter knows how
to read that frontmatter, fingerprint the structural shapes an edit must preserve,
and run Mintlify's own broken-link checker over a patched tree.

It is docsync's default adapter; `DocusaurusAdapter` and `MarkdownAdapter` cover the
other shipped frameworks, and `base.DocAdapter` is the seam for the next one.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import frontmatter

from docsync.adapters.base import DocAdapter

# Mintlify v4 reads `docs.json` and ignores the legacy `mint.json`; `docs.json`
# is therefore the canonical nav target. We mirror writes into `mint.json` when
# it exists (its shape differs) so the two don't silently diverge in a repo that
# still ships both.
_DOCS_JSON = "docs.json"
_MINT_JSON = "mint.json"

# MDX component opening tags we count. Restricting to a known set keeps the
# signature stable against stray capitalised prose words while still covering the
# components Mintlify pages actually use.
_KNOWN_COMPONENTS = frozenset(
    {
        "CardGroup",
        "Card",
        "Warning",
        "Note",
        "Tip",
        "Info",
        "Steps",
        "Step",
        "Accordion",
        "AccordionGroup",
        "Tabs",
        "Tab",
        "CodeGroup",
        "Frame",
    }
)

# Leaf components an edit may *add* when the diff motivates it (documenting a new
# field/route/stage often means one more <Step>/<Note>/<Card>). Their structural
# containers (CardGroup/Steps/Tabs/AccordionGroup/CodeGroup/Frame) stay frozen — adding a
# whole container is a real structural change — and *removing* any of these is always
# rejected (a count decrease is a deletion). Consumed by validate via the adapter hook.
_ADDITIVE_SAFE_COMPONENTS = frozenset(
    {"Card", "Warning", "Note", "Tip", "Info", "Step", "Accordion", "Tab"}
)

# A code-fence marker is a line whose first non-space content is ``` (3+ backticks).
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
# A mermaid fence opener: ```mermaid (optionally with trailing space).
_MERMAID_RE = re.compile(r"^\s*```mermaid\b", re.MULTILINE)

# Component tag in document order: opening `<Card ...>`, self-closing `<Card/>`, or
# closing `</Card>`. Groups: (1) leading slash, (2) name, (3) trailing slash.
_TAG_RE = re.compile(r"<(/?)([A-Z][A-Za-z]*)\b[^>]*?(/?)>", re.DOTALL)
# Fenced code blocks and inline code spans — component tags inside them are example
# text, not real structure, and must be ignored by the well-formedness scan.
_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _code_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char ranges of fenced + inline code — regions to leave untouched."""
    spans = [m.span() for m in _FENCE_BLOCK_RE.finditer(text)]
    spans += [m.span() for m in _INLINE_CODE_RE.finditer(text)]
    return spans


def _pos_in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)

# Broken-link check tuning: be defensive, never let the doc pipeline hang on it.
_LINK_CHECK_TIMEOUT_S = 30
_LINK_PROBLEM_RE = re.compile(r"broken|not found|404|missing|unreachable", re.IGNORECASE)
# Success/negation lines that contain a problem keyword but report the ABSENCE of
# problems (e.g. Mintlify's "success no broken links found"). These must not be
# mistaken for findings just because they say "broken".
_LINK_OK_RE = re.compile(
    r"no broken|0 broken|success|all links|no issues|no problems|✓|✔", re.IGNORECASE
)
# A real broken-link finding names the offending target (a URL, a path, or a doc
# file); status/progress lines like "checking for broken links..." do not. Requiring
# a target reference filters that tooling chatter without enumerating every phrase.
_LINK_REF_RE = re.compile(r"https?://|/|\.mdx?\b", re.IGNORECASE)
# Output that means the CLI itself never ran (package absent, npx aborted, ...).
# Such lines must NOT be mistaken for broken-link findings.
_CLI_UNAVAILABLE_RE = re.compile(
    r"npm error|npx canceled|command not found|could not determine executable"
    r"|is not recognized|need to install",
    re.IGNORECASE,
)


class MintlifyAdapter(DocAdapter):
    """Adapter for Mintlify-flavoured MDX docs."""

    name = "mintlify"

    # -- ownership ---------------------------------------------------------

    def owns(self, page_path: str) -> bool:
        """True for `.mdx` and `.md` files (case-insensitive)."""
        lowered = page_path.lower()
        return lowered.endswith(".mdx") or lowered.endswith(".md")

    # -- frontmatter -------------------------------------------------------

    def split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Return (metadata, body). Empty dict + original text if no frontmatter.

        Uses the `frontmatter` library; on any parse error we degrade to treating
        the whole text as the body (no frozen keys to compare), so a malformed
        edit surfaces elsewhere rather than crashing the splitter.
        """
        try:
            post = frontmatter.loads(text)
        except Exception:
            return {}, text
        # `metadata` is a dict (possibly empty); `content` is the body without
        # the frontmatter block.
        return dict(post.metadata), post.content

    def frontmatter_keys_to_freeze(self) -> list[str]:
        return ["title", "description"]

    # -- authoring / additive-edit policy ---------------------------------

    def additive_safe_components(self) -> frozenset[str]:
        """Leaf MDX components an edit may add (containers stay frozen)."""
        return _ADDITIVE_SAFE_COMPONENTS

    def authoring_components_hint(self) -> str:
        return (
            "This page uses Mintlify MDX components. You MAY add new leaf components — "
            "<Note>, <Tip>, <Warning>, <Info>, <Card>, <Step>, <Accordion>, <Tab> — when "
            "the content warrants one, but ALWAYS as a balanced pair (every <Note> needs "
            "its </Note>; a self-closing <Card/> is fine). NEVER remove a component, alter "
            "a container component (<CardGroup>, <Steps>, <Tabs>, <AccordionGroup>), reorder "
            "or mis-nest tags, or change the number of code/mermaid fences."
        )

    # -- structural fingerprint -------------------------------------------

    def structural_signature(self, text: str) -> dict:
        """A flat {str: int} fingerprint of MDX structure to hold invariant.

        Keys:
          - one entry per known component open tag: `{"Card": n}`
          - one entry per known component close tag: `{"/Card": n}`
          - `fence_count`: number of ``` code-fence markers (must stay even)
          - `mermaid_count`: number of ```mermaid fences

        A self-closing `<Card/>` counts as BOTH an open and a close, so its open/close
        deltas stay balanced — matching `structural_problems` ("self-closing is balanced
        by definition"). Counting it as an open-only (the old behaviour) made the additive
        gate reject a benign self-closing addition as `opens +1, closes +0`.

        Flat int values so two signatures compare with a plain `==`.
        """
        signature: dict = {}

        for match in _TAG_RE.finditer(text):
            closing, name, self_closing = match.group(1), match.group(2), match.group(3)
            if name not in _KNOWN_COMPONENTS:
                continue
            if closing:
                signature[f"/{name}"] = signature.get(f"/{name}", 0) + 1
            else:
                signature[name] = signature.get(name, 0) + 1
                if self_closing:
                    signature[f"/{name}"] = signature.get(f"/{name}", 0) + 1

        signature["fence_count"] = len(_FENCE_RE.findall(text))
        signature["mermaid_count"] = len(_MERMAID_RE.findall(text))
        return signature

    # -- well-formedness (nesting / balance) ------------------------------

    def structural_problems(self, text: str) -> list[str]:
        """Stack-check known component tags for balance + correct nesting.

        Catches the failure modes the count signature misses: a dropped closing
        tag, a stray close, or a swapped/reordered pair (whose counts still match).
        Tags inside fenced code blocks / inline code are ignored (they're examples),
        and only known components are tracked (so arbitrary `<Thing>` prose can't
        produce false positives). Self-closing `<Card/>` is balanced by definition.
        """
        scan = _INLINE_CODE_RE.sub("", _FENCE_BLOCK_RE.sub("", text))
        stack: list[str] = []
        problems: list[str] = []

        for match in _TAG_RE.finditer(scan):
            closing, name, self_closing = match.group(1), match.group(2), match.group(3)
            if name not in _KNOWN_COMPONENTS:
                continue
            if self_closing:
                continue
            if not closing:
                stack.append(name)
                continue
            # A closing tag.
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                problems.append(
                    f"mismatched nesting: </{name}> closes <{stack[-1]}>"
                )
                while stack and stack.pop() != name:
                    pass
            else:
                problems.append(f"stray closing </{name}> with no matching open tag")

        problems.extend(f"unclosed <{name}> (no matching close tag)" for name in stack)
        return problems

    def repair_structure(self, text: str) -> str:
        """Close trailing unclosed components and drop stray closers (safe repairs).

        Mirrors `structural_problems`' stack walk but rewrites the text:
          - an unclosed `<Steps>` left on the stack → append `</Steps>` at the end;
          - a stray `</Note>` matching no open tag → delete that tag.
        Mis-nesting (a closer that matches an *inner* open) is ambiguous, so we bail
        and return the text unchanged for the validator to reject. Tags inside code
        are never touched. Idempotent: repairing well-formed text is a no-op.
        """
        spans = _code_spans(text)
        stack: list[str] = []
        strays: list[tuple[int, int]] = []  # (start, end) of stray closers to delete

        for match in _TAG_RE.finditer(text):
            if _pos_in_spans(match.start(), spans):
                continue
            closing, name, self_closing = match.group(1), match.group(2), match.group(3)
            if name not in _KNOWN_COMPONENTS or self_closing:
                continue
            if not closing:
                stack.append(name)
            elif stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                return text  # mis-nesting — too risky to auto-fix
            else:
                strays.append(match.span())

        if not stack and not strays:
            return text  # already well-formed — nothing to do

        out = text
        for start, end in sorted(strays, reverse=True):  # right-to-left keeps offsets
            out = out[:start] + out[end:]
        if stack:
            closers = "\n".join(f"</{name}>" for name in reversed(stack))
            out = out.rstrip() + "\n\n" + closers + "\n"
        return out

    # -- link check (soft gate) -------------------------------------------

    def check_links(self, docs_root: Path) -> list[str]:
        """Run Mintlify's broken-link checker over `docs_root`.

        Returns human-readable problem lines (empty == clean / unavailable). This
        must NEVER raise: a missing CLI, a timeout, or a non-zero exit are all
        treated as "no findings" so validation can downgrade it to a warning.
        """
        # Try `npx --no-install mintlify broken-links` first (project-local CLI),
        # then a bare `mintlify broken-links` (globally installed CLI).
        command_variants = (
            ["npx", "--no-install", "mintlify", "broken-links"],
            ["mintlify", "broken-links"],
        )

        for command in command_variants:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(docs_root),
                    capture_output=True,
                    text=True,
                    timeout=_LINK_CHECK_TIMEOUT_S,
                    check=False,
                )
            except (
                FileNotFoundError,
                subprocess.TimeoutExpired,
                OSError,
                ValueError,
            ):
                # CLI not present / hung / bad invocation — try the next variant.
                continue
            except Exception:
                # Absolutely never let a link check sink the pipeline.
                continue

            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            # If the CLI never actually ran (package not installed, npx aborted),
            # treat it as unavailable and try the next variant rather than parsing
            # tooling noise as broken-link findings.
            if _CLI_UNAVAILABLE_RE.search(output):
                continue
            return self._parse_link_output(output)

        return []

    @staticmethod
    def _parse_link_output(output: str) -> list[str]:
        """Pull broken-link lines out of the CLI's stdout/stderr, defensively."""
        problems: list[str] = []
        try:
            for raw_line in output.splitlines():
                line = raw_line.strip()
                if (
                    line
                    and _LINK_PROBLEM_RE.search(line)
                    and _LINK_REF_RE.search(line)
                    and not _LINK_OK_RE.search(line)
                ):
                    problems.append(line)
        except Exception:
            return []
        return problems

    # -- navigation (bootstrap: register new pages) ------------------------

    @staticmethod
    def _page_ref(page_path: str) -> str:
        """A Mintlify nav route: the page path minus its extension, no leading slash."""
        ref = page_path[:-4] if page_path.lower().endswith(".mdx") else (
            page_path[:-3] if page_path.lower().endswith(".md") else page_path
        )
        return ref.lstrip("/")

    @staticmethod
    def _groups_list(data: dict) -> list:
        """Return the mutable list of {group, pages} entries for either nav shape.

        `docs.json` nests them under `navigation.groups`; legacy `mint.json` puts
        them directly in `navigation`. A missing nav is initialized in the
        `docs.json` shape. Returns the list object held inside `data`.
        """
        nav = data.get("navigation")
        if isinstance(nav, dict):
            groups = nav.setdefault("groups", [])
            return groups
        if isinstance(nav, list):
            return nav
        data["navigation"] = {"groups": []}
        return data["navigation"]["groups"]

    def nav_routes(self, docs_root: Path) -> list[str]:
        """Every page ref currently in docs.json (falling back to mint.json)."""
        for name in (_DOCS_JSON, _MINT_JSON):
            path = Path(docs_root) / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            routes: list[str] = []
            for grp in self._groups_list(data):
                if isinstance(grp, dict):
                    routes.extend(p for p in grp.get("pages", []) if isinstance(p, str))
            return routes
        return []

    def register_pages_in_nav(
        self, docs_root: Path, page_paths: list[str], *, group: str
    ) -> list[str]:
        """Add pages to the `group` nav group in docs.json (mirrored to mint.json).

        Idempotent: a page already present anywhere in the nav is skipped, and the
        target group is created only if absent. Returns the nav files modified.
        """
        refs = [self._page_ref(p) for p in page_paths]
        modified: list[str] = []
        for name in (_DOCS_JSON, _MINT_JSON):
            path = Path(docs_root) / name
            if not path.exists():
                continue
            if self._register_in_file(path, refs, group):
                modified.append(name)
        return modified

    @staticmethod
    def _detect_indent(text: str) -> int:
        """Infer a JSON file's indent width from its first indented line (default 2)."""
        for line in text.splitlines():
            stripped = line.lstrip(" ")
            if stripped and stripped != line:
                return len(line) - len(stripped)
        return 2

    def set_nav_sections(
        self, docs_root: Path, ordered_sections: list[tuple[str, list[str]]]
    ) -> list[str]:
        """Write nav groups in the given order; refs are already extensionless routes."""
        modified: list[str] = []
        for name in (_DOCS_JSON, _MINT_JSON):
            path = Path(docs_root) / name
            if not path.exists():
                continue
            if self._apply_sections(path, ordered_sections):
                modified.append(name)
        return modified

    def _apply_sections(
        self, path: Path, ordered_sections: list[tuple[str, list[str]]]
    ) -> bool:
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return False
        indent = self._detect_indent(raw)
        groups = self._groups_list(data)

        # Every route already in the nav (any group) — for cross-group set-union.
        existing: set[str] = set()
        for grp in groups:
            if isinstance(grp, dict):
                existing.update(p for p in grp.get("pages", []) if isinstance(p, str))

        changed = False
        for section, routes in ordered_sections:
            target = next(
                (g for g in groups if isinstance(g, dict) and g.get("group") == section),
                None,
            )
            if target is None:
                target = {"group": section, "pages": []}
                groups.append(target)  # new sections append in reading-flow order
                changed = True
            for ref in routes:
                if ref not in existing:
                    target.setdefault("pages", []).append(ref)
                    existing.add(ref)
                    changed = True

        if changed:
            path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
        return changed

    def ensure_valid_docs_json(self, docs_root: Path) -> bool:
        """Create/repair docs.json so an empty scaffold renders (e.g. required colors)."""
        path = Path(docs_root) / _DOCS_JSON
        defaults = {
            "$schema": "https://mintlify.com/docs.json",
            "theme": "mint",
            "name": "Documentation",
            "colors": {"primary": "#16A34A", "light": "#16A34A", "dark": "#16A34A"},
            "navigation": {"groups": []},
        }
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
            return True
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return False
        indent = self._detect_indent(raw)
        changed = False
        for key, value in defaults.items():
            if key == "$schema":
                continue
            if key not in data:
                data[key] = value
                changed = True
        if changed:
            path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
        return changed

    def _register_in_file(self, path: Path, refs: list[str], group: str) -> bool:
        """Set-union `refs` into `group` within one nav file. True if it changed."""
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return False
        indent = self._detect_indent(raw)

        groups = self._groups_list(data)
        existing: set[str] = set()
        for grp in groups:
            if isinstance(grp, dict):
                existing.update(p for p in grp.get("pages", []) if isinstance(p, str))
        new_refs = [r for r in refs if r not in existing]
        if not new_refs:
            return False  # idempotent no-op

        target = next(
            (g for g in groups if isinstance(g, dict) and g.get("group") == group), None
        )
        if target is None:
            target = {"group": group, "pages": []}
            groups.append(target)
        target.setdefault("pages", []).extend(new_refs)

        # Preserve the file's existing indent so the mirror produces a minimal diff
        # (docs.json is 2-space, legacy mint.json is often 4-space).
        path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
        return True
