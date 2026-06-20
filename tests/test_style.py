"""Tests for the shared documentation-craft constants (`docsync.style`).

These are plain prompt fragments, so the contract is simply: they're non-empty, the
per-kind structures are distinct, and the lookup defaults to reference for unknown kinds.
"""

from __future__ import annotations

from docsync import style


def test_principle_constants_present():
    for const in (style.INVERTED_PYRAMID, style.SCANNABILITY, style.GROUNDING,
                  style.DIATAXIS_DISCIPLINE):
        assert isinstance(const, str) and len(const) > 40


def test_kind_structure_distinct_per_kind():
    ref = style.kind_structure("reference")
    guide = style.kind_structure("guide")
    concept = style.kind_structure("concept")
    assert "REFERENCE" in ref and "GUIDE" in guide and "CONCEPT" in concept
    assert len({ref, guide, concept}) == 3


def test_kind_structure_defaults_to_reference():
    assert style.kind_structure("nonsense") == style.kind_structure("reference")
