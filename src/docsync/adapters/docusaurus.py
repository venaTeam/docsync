"""Docusaurus doc-framework adapter.

Docusaurus pages are `.md`/`.mdx` files with YAML frontmatter. Their structural chrome
is **admonitions** — `:::note … :::` block directives — plus a handful of MDX components
(`<Tabs>`/`<TabItem>`, `<Details>`) on `.mdx` pages. This adapter fingerprints those
shapes, runs a self-contained on-disk broken-link check (no external CLI — the whole
point for air-gapped installs, where Mintlify's hosted `npx mintlify` checker can't run),
and maintains a manual `sidebars.js` for bootstrap navigation.

Navigation note: Docusaurus reads its sidebar from `sidebars.js` (or `sidebars.ts`) at the
**project root** — the directory holding `docusaurus.config.{js,ts}` — which is the parent
of the docs content dir, not the content dir itself. The nav helpers locate that root by
walking up from `docs_root`. This adapter assumes an already-initialized Docusaurus project
(it does not fabricate `docusaurus.config.js`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import frontmatter

from docsync.adapters.base import DocAdapter

# -- admonitions -----------------------------------------------------------

# The standard Docusaurus admonition keywords. Restricting to a known set keeps the
# signature stable against an arbitrary `:::foo` that isn't a real admonition.
_KNOWN_ADMONITIONS = frozenset({"note", "tip", "info", "warning", "danger", "caution"})

# An admonition opener: a line starting with 3+ colons followed by a keyword
# (`:::note`, `:::tip Title`, `:::note[Title]`). A closer is a bare `:::` line.
_ADMONITION_OPEN_RE = re.compile(r"^[ \t]*:{3,}[ \t]*([a-zA-Z]+)")
_ADMONITION_CLOSE_RE = re.compile(r"^[ \t]*:{3,}[ \t]*$")

# -- code fences -----------------------------------------------------------

# A code-fence marker: a line whose first non-space content is ``` (3+ backticks).
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_MERMAID_RE = re.compile(r"^\s*```mermaid\b", re.MULTILINE)
# Same pattern used with `.match` to toggle in/out of a fenced block line-by-line.
_FENCE_TOGGLE_RE = re.compile(r"[ \t]*```")

# -- MDX components (on `.mdx` pages) --------------------------------------

# Component tag in document order: opening `<Tabs ...>`, self-closing `<Tabs/>`, or
# closing `</Tabs>`. Groups: (1) leading slash, (2) name, (3) trailing slash.
_TAG_RE = re.compile(r"<(/?)([A-Z][A-Za-z]*)\b[^>]*?(/?)>", re.DOTALL)
_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
# The Docusaurus MDX components worth tracking. `Tabs`/`Details` are containers (frozen);
# `TabItem` is the addable leaf.
_KNOWN_COMPONENTS = frozenset({"Tabs", "TabItem", "Details"})

# -- links -----------------------------------------------------------------

# A Markdown inline link: `[text](target)`. We only care about the target.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)")
# Page extensions the on-disk checker resolves a relative link against.
_PAGE_SUFFIXES = (".md", ".mdx", ".markdown")
# A URL scheme like `http:`, `mailto:`, `tel:` — an external target we never check.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

# -- navigation ------------------------------------------------------------

_SIDEBARS_JS = "sidebars.js"
_SIDEBARS_TS = "sidebars.ts"
_DOCUSAURUS_CONFIG = ("docusaurus.config.js", "docusaurus.config.ts")
# Default sidebar id we author into when no manual sidebar exists yet.
_DEFAULT_SIDEBAR_KEY = "docsSidebar"
# Docusaurus's default numberPrefixParser: `01-intro.md` gets doc id `intro`, and the
# stripping applies to every path segment (directories too). Sidebar entries must use
# the stripped id or the build fails with an unknown-document error.
_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*[-_.]+\s*(.+)$")


def _code_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char ranges of fenced + inline code — regions to leave untouched."""
    spans = [m.span() for m in _FENCE_BLOCK_RE.finditer(text)]
    spans += [m.span() for m in _INLINE_CODE_RE.finditer(text)]
    return spans


def _pos_in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _admonition_events(text: str) -> list[tuple[str, str | None]]:
    """Admonition opens/closes in document order, ignoring fenced code.

    Returns `("open", kind)` for **every** `:::kind` opener — known or custom — and
    `("close", None)` for a bare `:::` closer. Custom keywords are a supported
    Docusaurus config (`admonitions.keywords`), so they must participate in the
    balance stack (else their closer reads as stray); consumers fingerprint only
    `_KNOWN_ADMONITIONS`. A line walk (not a regex over the whole text) so the
    fence-state toggle is reliable — admonitions are line-based directives.
    """
    events: list[tuple[str, str | None]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_TOGGLE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        opener = _ADMONITION_OPEN_RE.match(line)
        if opener:
            events.append(("open", opener.group(1).lower()))
            continue
        if _ADMONITION_CLOSE_RE.match(line):
            events.append(("close", None))
    return events


class DocusaurusAdapter(DocAdapter):
    """Adapter for Docusaurus Markdown/MDX docs (admonitions + sidebars.js nav)."""

    name = "docusaurus"
    # Admonitions render in plain `.md`; we author there rather than `.mdx` unless a page
    # needs JSX. `owns` still accepts `.mdx`.
    page_extension = ".md"

    # -- ownership ---------------------------------------------------------

    def owns(self, page_path: str) -> bool:
        """True for `.md`, `.mdx`, and `.markdown` files (case-insensitive)."""
        lowered = page_path.lower()
        return lowered.endswith((".md", ".mdx", ".markdown"))

    # -- frontmatter -------------------------------------------------------

    def split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Return (metadata, body). Empty dict + original text on any parse error."""
        try:
            post = frontmatter.loads(text)
        except Exception:
            return {}, text
        return dict(post.metadata), post.content

    def frontmatter_keys_to_freeze(self) -> list[str]:
        return ["title", "description"]

    # -- authoring / additive-edit policy ---------------------------------

    def additive_safe_components(self) -> frozenset[str]:
        """Admonition types + `TabItem` may be added; `Tabs`/`Details` stay frozen."""
        return _KNOWN_ADMONITIONS | {"TabItem"}

    def authoring_components_hint(self) -> str:
        return (
            "This page uses Docusaurus Markdown. For callouts use admonitions — :::note, "
            ":::tip, :::warning, :::info, :::danger — each opened with a `:::<type>` line "
            "and closed with a bare `:::` line of its own; reserve one for the single "
            "caveat that matters. In .mdx pages you MAY use <Tabs>/<TabItem>. You MAY add "
            "a new admonition when the content warrants one, but ALWAYS balanced (every "
            "`:::` opener needs its closing `:::`). NEVER remove a callout, reorder or "
            "mis-nest blocks, or change the number of code/mermaid fences."
        )

    # -- structural fingerprint -------------------------------------------

    def structural_signature(self, text: str) -> dict:
        """A flat {str: int} fingerprint of Docusaurus structure to hold invariant.

        Keys:
          - one entry per admonition type open: `{"note": n}`; each bare `:::` closer is
            attributed (via a stack) to the type it closes: `{"/note": n}`. Matched
            open/close keys let `validate._check_structure`'s additive carve-out treat a
            new balanced admonition like a new balanced MDX component.
          - one entry per known MDX component open/close (`Tabs`/`TabItem`/`Details`).
          - `fence_count`: number of ``` markers (must stay even); `mermaid_count`.
        """
        signature: dict = {}

        stack: list[str] = []
        for event, kind in _admonition_events(text):
            if event == "open" and kind is not None:
                # Custom keywords ride the stack (so closer attribution stays
                # positional) but are not fingerprinted — symmetric with their
                # closer below, so a custom block leaves the signature untouched.
                if kind in _KNOWN_ADMONITIONS:
                    signature[kind] = signature.get(kind, 0) + 1
                stack.append(kind)
            else:  # a closer — attribute it to the open it balances
                closed = stack.pop() if stack else "_unmatched"
                if closed in _KNOWN_ADMONITIONS or closed == "_unmatched":
                    signature[f"/{closed}"] = signature.get(f"/{closed}", 0) + 1

        scan = _INLINE_CODE_RE.sub("", _FENCE_BLOCK_RE.sub("", text))
        for match in _TAG_RE.finditer(scan):
            closing, cname, self_closing = match.group(1), match.group(2), match.group(3)
            if cname not in _KNOWN_COMPONENTS:
                continue
            if closing:
                signature[f"/{cname}"] = signature.get(f"/{cname}", 0) + 1
            else:
                signature[cname] = signature.get(cname, 0) + 1
                if self_closing:
                    signature[f"/{cname}"] = signature.get(f"/{cname}", 0) + 1

        signature["fence_count"] = len(_FENCE_RE.findall(text))
        signature["mermaid_count"] = len(_MERMAID_RE.findall(text))
        return signature

    # -- well-formedness (nesting / balance) ------------------------------

    def structural_problems(self, text: str) -> list[str]:
        """Stack-check admonitions and MDX components for balance.

        Admonitions and components are checked with independent stacks (a Docusaurus
        admonition is a Markdown directive, a component is JSX — they don't share one
        nesting discipline). Tags inside code are ignored. Catches a dropped `:::`
        closer, a stray closer, or an unbalanced/mis-nested component pair.
        """
        problems: list[str] = []

        stack: list[str] = []
        for event, kind in _admonition_events(text):
            if event == "open":
                stack.append(kind or "")
            elif stack:
                stack.pop()
            else:
                problems.append("stray closing ::: with no matching admonition")
        problems.extend(f"unclosed :::{kind} (no matching ::: close)" for kind in stack)

        scan = _INLINE_CODE_RE.sub("", _FENCE_BLOCK_RE.sub("", text))
        cstack: list[str] = []
        for match in _TAG_RE.finditer(scan):
            closing, name, self_closing = match.group(1), match.group(2), match.group(3)
            if name not in _KNOWN_COMPONENTS or self_closing:
                continue
            if not closing:
                cstack.append(name)
                continue
            if cstack and cstack[-1] == name:
                cstack.pop()
            elif name in cstack:
                problems.append(f"mismatched nesting: </{name}> closes <{cstack[-1]}>")
                while cstack and cstack.pop() != name:
                    pass
            else:
                problems.append(f"stray closing </{name}> with no matching open tag")
        problems.extend(f"unclosed <{name}> (no matching close tag)" for name in cstack)
        return problems

    def repair_structure(self, text: str) -> str:
        """Close trailing unclosed admonitions / components and drop stray closers.

        Mirrors `structural_problems`' two stack walks but rewrites the text with only
        the unambiguous, low-risk fixes; mis-nested components bail (return unchanged)
        for the validator to reject. Idempotent on well-formed text. Always re-validated
        by the caller, so a bad repair can never ship.
        """
        out = self._repair_components(text)
        if out is None:
            return text  # ambiguous component mis-nesting — leave for the validator
        return self._repair_admonitions(out)

    @staticmethod
    def _repair_components(text: str) -> str | None:
        """Close trailing components, drop stray closers; None on mis-nesting."""
        spans = _code_spans(text)
        stack: list[str] = []
        strays: list[tuple[int, int]] = []
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
                return None
            else:
                strays.append(match.span())

        out = text
        for start, end in sorted(strays, reverse=True):  # right-to-left keeps offsets
            out = out[:start] + out[end:]
        if stack:
            closers = "\n".join(f"</{name}>" for name in reversed(stack))
            out = out.rstrip() + "\n\n" + closers + "\n"
        return out

    @staticmethod
    def _repair_admonitions(text: str) -> str:
        """Append a `:::` for each unclosed admonition; drop stray bare `:::` closers."""
        lines = text.splitlines(keepends=True)
        in_fence = False
        depth = 0
        kept: list[str] = []
        for line in lines:
            if _FENCE_TOGGLE_RE.match(line):
                in_fence = not in_fence
                kept.append(line)
                continue
            if not in_fence and _ADMONITION_OPEN_RE.match(line):
                # Any keyword opener — custom kinds too, so their closer is kept.
                depth += 1
            elif not in_fence and _ADMONITION_CLOSE_RE.match(line):
                if depth == 0:
                    continue  # stray closer — drop it
                depth -= 1
            kept.append(line)

        out = "".join(kept)
        if depth:
            out = out.rstrip() + "\n\n" + "\n".join([":::"] * depth) + "\n"
        return out

    # -- link check (soft gate) -------------------------------------------

    def check_links(self, docs_root: Path) -> list[str]:
        """Self-contained on-disk broken-link check — no external CLI, never raises.

        Walks every page under `docs_root`, pulls Markdown `[text](target)` links, and
        flags a **relative file link** whose target resolves to nothing on disk. To keep
        a soft gate quiet, only clearly-checkable links are inspected — a target that
        ends in a page extension, or starts with `./`/`../`. External URLs, in-page
        anchors, absolute `/route` slugs (resolved by route, not path), and bare doc-id
        words are skipped. Returns human-readable problem lines (empty == clean).
        """
        root = Path(docs_root)
        problems: list[str] = []
        try:
            pages = [
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in _PAGE_SUFFIXES
            ]
        except OSError:
            return []

        for page in pages:
            try:
                text = page.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            scan = _INLINE_CODE_RE.sub("", _FENCE_BLOCK_RE.sub("", text))
            for match in _MD_LINK_RE.finditer(scan):
                target = match.group(1).split("#", 1)[0].strip()
                if not self._is_checkable_link(target):
                    continue
                if not self._relative_target_exists(page, target):
                    try:
                        where = page.relative_to(root)
                    except ValueError:
                        where = page
                    problems.append(f"{where}: broken link to {match.group(1)}")
        return problems

    @staticmethod
    def _is_checkable_link(target: str) -> bool:
        """Only inspect clearly-relative file links — skip the ambiguous rest."""
        if not target or target.startswith(("#", "/")):
            return False  # anchor or absolute route slug
        if target.startswith("//") or _SCHEME_RE.match(target):
            return False  # protocol-relative or scheme (http, mailto, …)
        lowered = target.lower()
        return lowered.endswith(_PAGE_SUFFIXES) or target.startswith(("./", "../"))

    @staticmethod
    def _relative_target_exists(page: Path, target: str) -> bool:
        """True if `target` (relative to `page`'s dir) resolves to a file on disk."""
        base = page.parent
        candidates = [base / target]
        if not target.lower().endswith(_PAGE_SUFFIXES):
            candidates += [base / f"{target}{suffix}" for suffix in _PAGE_SUFFIXES]
            candidates += [base / target / f"index{suffix}" for suffix in _PAGE_SUFFIXES]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return True
            except OSError:
                continue
        return False

    # -- navigation (bootstrap: sidebars.js) -------------------------------

    @staticmethod
    def _route_of(page_path: str) -> str:
        """A Docusaurus doc id: extension dropped, number prefixes stripped per segment."""
        stem = page_path
        lowered = page_path.lower()
        for suffix in (".markdown", ".mdx", ".md"):
            if lowered.endswith(suffix):
                stem = page_path[: -len(suffix)]
                break
        segments = [
            (match.group(1) if (match := _NUMBER_PREFIX_RE.match(seg)) else seg)
            for seg in stem.lstrip("/").split("/")
        ]
        return "/".join(segments)

    @staticmethod
    def _project_root(docs_root: Path) -> Path:
        """The Docusaurus project root (dir holding docusaurus.config.*).

        Walk up from the docs content dir; fall back to its parent (the normal layout,
        `<root>/docs/`) when no config is found.
        """
        root = Path(docs_root)
        for candidate in (root, *list(root.parents)[:4]):
            if any((candidate / name).exists() for name in _DOCUSAURUS_CONFIG):
                return candidate
        return root.parent if root.parent != root else root

    def _sidebars_path(self, docs_root: Path) -> Path:
        """Path to the sidebar file (existing `.ts` honored, else `sidebars.js`)."""
        project = self._project_root(docs_root)
        ts = project / _SIDEBARS_TS
        if ts.exists():
            return ts
        return project / _SIDEBARS_JS

    @staticmethod
    def _rel_to_docs_root(docs_root: Path, path: Path) -> str:
        """The nav file path relative to docs_root (bootstrap prefixes + normalizes it)."""
        return os.path.relpath(path, Path(docs_root)).replace(os.sep, "/")

    @staticmethod
    def _object_literal(raw: str) -> str | None:
        """The `{...}` object body of a sidebars module (first `{` to last `}`)."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return raw[start : end + 1]

    def _read_sidebars(self, path: Path) -> tuple[str, dict] | None:
        """Parse a sidebar file we wrote (JSON-shaped body) into (sidebar_key, data)."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        body = self._object_literal(raw)
        if body is None:
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not data:
            return None
        key = next(iter(data))
        sidebar = data[key]
        if not isinstance(sidebar, list):
            return None
        return key, data

    @staticmethod
    def _collect_doc_ids(node, out: list[str]) -> None:
        """Recursively gather doc-id strings from a sidebar list/category tree."""
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, list):
            for child in node:
                DocusaurusAdapter._collect_doc_ids(child, out)
        elif isinstance(node, dict):
            if node.get("type") == "doc" and isinstance(node.get("id"), str):
                out.append(node["id"])
            if "items" in node:
                DocusaurusAdapter._collect_doc_ids(node["items"], out)

    def nav_routes(self, docs_root: Path) -> list[str]:
        """Every doc id referenced by **any** sidebar in the file (empty if unreadable)."""
        path = self._sidebars_path(docs_root)
        if not path.exists():
            return []
        parsed = self._read_sidebars(path)
        if parsed is None:
            return []
        _key, data = parsed
        routes: list[str] = []
        for sidebar in data.values():
            self._collect_doc_ids(sidebar, routes)
        return routes

    def _write_sidebars(self, path: Path, key: str, data: dict) -> None:
        """Serialize the sidebar object as a JS (or TS) module Docusaurus can import."""
        body = json.dumps(data, indent=2)
        if path.name.endswith(".ts"):
            text = (
                "import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';\n\n"
                f"const sidebars: SidebarsConfig = {body};\n\nexport default sidebars;\n"
            )
        else:
            text = f"// @ts-check\nmodule.exports = {body};\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _apply_sections(
        self, docs_root: Path, ordered_sections: list[tuple[str, list[str]]]
    ) -> list[str]:
        """Set-union ordered category→items into the sidebar. Returns files modified."""
        path = self._sidebars_path(docs_root)
        if path.exists():
            parsed = self._read_sidebars(path)
            if parsed is None:
                # An existing sidebar we can't round-trip (hand-written JS, comments,
                # `require(...)`): refuse to clobber it — leave nav to the operator.
                return []
            key, data = parsed
        else:
            key, data = _DEFAULT_SIDEBAR_KEY, {_DEFAULT_SIDEBAR_KEY: []}
        groups = data[key]

        # Dedupe against every sidebar in the file, not just the one we author into —
        # a page already navigable from a second sidebar must not be re-registered.
        acc: list[str] = []
        for sidebar in data.values():
            self._collect_doc_ids(sidebar, acc)
        existing = set(acc)

        changed = False
        for section, routes in ordered_sections:
            target = next(
                (g for sidebar in data.values() if isinstance(sidebar, list)
                 for g in sidebar
                 if isinstance(g, dict) and g.get("type") == "category"
                 and g.get("label") == section),
                None,
            )
            if target is None:
                target = {"type": "category", "label": section, "items": []}
                groups.append(target)
                changed = True
            for route in routes:
                if route not in existing:
                    target.setdefault("items", []).append(route)
                    existing.add(route)
                    changed = True

        if not changed:
            return []
        self._write_sidebars(path, key, data)
        return [self._rel_to_docs_root(docs_root, path)]

    def set_nav_sections(
        self, docs_root: Path, ordered_sections: list[tuple[str, list[str]]]
    ) -> list[str]:
        """Write category groups in reading-flow order; routes are extensionless ids."""
        return self._apply_sections(docs_root, ordered_sections)

    def register_pages_in_nav(
        self, docs_root: Path, page_paths: list[str], *, group: str
    ) -> list[str]:
        """Add pages to the `group` category in the sidebar (idempotent set-union)."""
        routes = [self._route_of(p) for p in page_paths]
        return self._apply_sections(docs_root, [(group, routes)])

    def ensure_valid_docs_json(self, docs_root: Path) -> bool:
        """Create an empty `sidebars.js` when none exists so the nav manifest is present.

        Does NOT fabricate `docusaurus.config.js` — bootstrap targets an initialized
        Docusaurus project. Returns True if a file was created.
        """
        path = self._sidebars_path(docs_root)
        if path.exists():
            return False
        self._write_sidebars(path, _DEFAULT_SIDEBAR_KEY, {_DEFAULT_SIDEBAR_KEY: []})
        return True
