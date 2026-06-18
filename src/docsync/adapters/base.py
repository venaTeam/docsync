"""Doc-framework adapter interface.

An adapter encapsulates everything format-specific about a docs framework:
how to read a page's frontmatter, what structural invariants must be preserved
in an edit, and how to run the framework's own link/build checks. The MVP ships
only `MintlifyAdapter`; this interface is the seam for Docusaurus/GitBook later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DocAdapter(ABC):
    name: str

    @abstractmethod
    def owns(self, page_path: str) -> bool:
        """True if this adapter handles the given page (by extension/location)."""

    @abstractmethod
    def split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Return (frontmatter_dict, body). Empty dict if no frontmatter."""

    @abstractmethod
    def frontmatter_keys_to_freeze(self) -> list[str]:
        """Frontmatter keys an edit must never change (e.g. title, description)."""

    @abstractmethod
    def structural_signature(self, text: str) -> dict:
        """A structural fingerprint (component tag counts, fence balance, ...).

        validate.py compares the signature before vs after an edit; any change
        in counts/nesting fails the component-integrity gate.
        """

    def structural_problems(self, text: str) -> list[str]:
        """Well-formedness problems in `text` (unbalanced / mis-nested components).

        Complements `structural_signature`: counts can stay equal while nesting
        breaks (a reordered or swapped tag pair), so this checks the *new* text is
        actually well-formed. Default: no checks (subclasses override). Returns a
        list of human-readable problems (empty == well-formed).
        """
        return []

    @abstractmethod
    def check_links(self, docs_root: Path) -> list[str]:
        """Run the framework's broken-link check over a patched tree.

        Returns a list of human-readable problems (empty == clean). Implementations
        should degrade gracefully (return []) when the framework CLI is unavailable,
        and surface that as a warning via the caller.
        """

    def nav_routes(self, docs_root: Path) -> list[str]:
        """Existing navigation routes (extensionless page paths), for collision checks.

        Used by bootstrap to avoid proposing a page that's already in the nav.
        Default: none (a framework with no nav file has nothing to collide with).
        """
        return []

    def register_pages_in_nav(
        self, docs_root: Path, page_paths: list[str], *, group: str
    ) -> list[str]:
        """Add `page_paths` (page paths under docs_root) to the framework's nav.

        Idempotent: re-registering an existing page is a no-op. Returns the list of
        nav files actually modified (empty == no nav concept / nothing changed).
        Default: no-op for frameworks without a navigation manifest.
        """
        return []

    def set_nav_sections(
        self, docs_root: Path, ordered_sections: list[tuple[str, list[str]]]
    ) -> list[str]:
        """Register several nav sections in reading-flow order.

        `ordered_sections` is `[(section_title, [route, ...]), ...]` in the desired
        sequence; `route` is an extensionless page route. Idempotent (set-union per
        section). Returns nav files modified. Default: no-op.
        """
        return []

    def ensure_valid_docs_json(self, docs_root: Path) -> bool:
        """Ensure the framework's nav/config file exists and has its required fields.

        Used when bootstrapping into an empty scaffold so the site renders. Returns
        True if anything was created/changed. Default: no-op for frameworks with no
        config file.
        """
        return False
