"""Doc-framework adapters.

`base.DocAdapter` is the seam; concrete adapters register their `name` in `ADAPTERS`
below. `make_adapter(name)` is the single resolution point — `validate.get_adapter`
(page-level), `bootstrap`, and `infer` all go through it, so adding a framework is one
entry here plus the adapter module. Config selects the active adapter via
`DocsyncConfig.adapter` (default `"mintlify"`).
"""

from __future__ import annotations

from .base import DocAdapter
from .markdown import MarkdownAdapter
from .mintlify import MintlifyAdapter

# name -> adapter class. The key is what a user writes as `adapter:` in config.yml.
ADAPTERS: dict[str, type[DocAdapter]] = {
    MintlifyAdapter.name: MintlifyAdapter,
    MarkdownAdapter.name: MarkdownAdapter,
}

DEFAULT_ADAPTER = MintlifyAdapter.name


def make_adapter(name: str) -> DocAdapter:
    """Instantiate the adapter registered under `name`.

    Raises ValueError on an unknown name — a typo'd `adapter:` should fail loudly at
    config-load/use time, not silently fall back to a different framework.
    """
    try:
        return ADAPTERS[name]()
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"unknown adapter {name!r} (known: {known})") from None


__all__ = ["ADAPTERS", "DEFAULT_ADAPTER", "DocAdapter", "MarkdownAdapter", "MintlifyAdapter",
           "make_adapter"]
