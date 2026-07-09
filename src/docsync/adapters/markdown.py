"""Plain-Markdown doc-framework adapter.

For docs sites that are ordinary `.md` files with YAML frontmatter and **no** inline
JSX/MDX components — e.g. a flat Markdown site, MkDocs, or a Docusaurus tree consumed
as plain Markdown. It freezes the same `title`/`description` frontmatter, fingerprints
only the code-fence balance (there are no components to track), and has no framework
link checker or navigation manifest (both degrade to no-ops).

This is the second adapter; it proves the `base.DocAdapter` seam beyond Mintlify. A
fuller Docusaurus adapter (sidebars.js nav, admonitions) can subclass or replace it.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from docsync.adapters.base import DocAdapter

# A code-fence marker: a line whose first non-space content is ``` (3+ backticks).
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_MERMAID_RE = re.compile(r"^\s*```mermaid\b", re.MULTILINE)


class MarkdownAdapter(DocAdapter):
    """Adapter for plain `.md` docs with YAML frontmatter (no MDX components)."""

    name = "markdown"
    page_extension = ".md"

    # -- ownership ---------------------------------------------------------

    def owns(self, page_path: str) -> bool:
        """True for `.md` and `.markdown` files (case-insensitive).

        Note: `.mdx` is intentionally NOT owned here — an `.mdx` tree is Mintlify/MDX
        territory. The active adapter is chosen by config, not by ownership races.
        """
        lowered = page_path.lower()
        return lowered.endswith(".md") or lowered.endswith(".markdown")

    # -- frontmatter -------------------------------------------------------

    def split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Return (metadata, body). Empty dict + original text if none / unparseable.

        Mirrors the Mintlify adapter: degrade to "all body" on any parse error so a
        malformed edit surfaces in a gate, not as a splitter crash.
        """
        try:
            post = frontmatter.loads(text)
        except Exception:
            return {}, text
        return dict(post.metadata), post.content

    def frontmatter_keys_to_freeze(self) -> list[str]:
        """Return the frontmatter keys whose values must stay frozen across edits.

        Returns:
            The keys ``title`` and ``description``, which edits may not change.
        """
        return ["title", "description"]

    # -- structural fingerprint -------------------------------------------

    def structural_signature(self, text: str) -> dict:
        """Plain Markdown has no components — fingerprint only fence balance.

        `fence_count` must stay even (an edit that opens a code block without closing
        it changes the parity and the component-integrity gate rejects it). `mermaid_count`
        tracks mermaid blocks the same way Mintlify does, so a dropped diagram is caught.
        """
        return {
            "fence_count": len(_FENCE_RE.findall(text)),
            "mermaid_count": len(_MERMAID_RE.findall(text)),
        }

    # structural_problems / repair_structure: defaults (no component nesting to check).

    # -- link check (soft gate) -------------------------------------------

    def check_links(self, docs_root: Path) -> list[str]:
        """No standard plain-Markdown link checker — always clean (no-op soft gate)."""
        return []

    # nav_routes / register_pages_in_nav / set_nav_sections / ensure_valid_docs_json:
    # plain Markdown has no navigation manifest, so the base no-ops apply. Bootstrap
    # therefore emits pages without touching a nav file (a flat Markdown site needs none).
