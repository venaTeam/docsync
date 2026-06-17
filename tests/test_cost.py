"""Cost-meter tests — pricing, accumulation, the metering proxy, and rendering.

No network / no real SDK: a fake client whose `messages.parse` returns a response
carrying a `usage` (object or dict) exercises the whole MeteredClient path.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from docsync import cost
from docsync.models import PageEdit


# --- pricing ----------------------------------------------------------------


def test_pricing_for_prefix_match():
    assert cost.pricing_for("claude-opus-4-8").output == 75.0
    assert cost.pricing_for("claude-haiku-4-5").input == 1.0
    assert cost.pricing_for("claude-sonnet-4-6").input == 3.0


def test_pricing_for_unknown_falls_back_to_opus_class():
    # Unknown → fallback (opus-class), so cost is never under-reported.
    assert cost.pricing_for("some-future-model").output == 75.0
    assert cost.pricing_for(None).output == 75.0


def test_cost_of_computes_per_token_class():
    usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    # 1M input @ $15 + 1M output @ $75 = $90 for opus.
    assert cost.cost_of("claude-opus-4-8", usage) == pytest.approx(90.0)


def test_cost_of_includes_cache_classes():
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 1_000_000,
             "cache_read_input_tokens": 1_000_000}
    # opus cache write $18.75 + cache read $1.50 per MTok.
    assert cost.cost_of("claude-opus-4-8", usage) == pytest.approx(20.25)


# --- _get tolerance ---------------------------------------------------------


def test_get_handles_object_dict_and_none():
    obj = SimpleNamespace(input_tokens=5)
    assert cost._get(obj, "input_tokens") == 5
    assert cost._get({"input_tokens": 7}, "input_tokens") == 7
    assert cost._get(None, "input_tokens") == 0
    assert cost._get({"input_tokens": None}, "input_tokens") == 0  # null → 0
    assert cost._get(obj, "missing") == 0


# --- meter accumulation -----------------------------------------------------


def _usage(i=0, o=0, cw=0, cr=0):
    return SimpleNamespace(
        input_tokens=i, output_tokens=o,
        cache_creation_input_tokens=cw, cache_read_input_tokens=cr,
    )


def test_meter_accumulates_per_model_and_totals():
    m = cost.UsageMeter()
    m.record("claude-opus-4-8", _usage(i=100, o=200))
    m.record("claude-opus-4-8", _usage(i=100, o=0, cr=50))
    m.record("claude-haiku-4-5", _usage(i=10, o=5))

    run = m.finalize()
    assert run.calls == 3
    assert run.input_tokens == 210
    assert run.output_tokens == 205
    assert run.cache_read_input_tokens == 50
    # by_model is cost-sorted: opus first (far pricier).
    assert run.by_model[0].model == "claude-opus-4-8"
    assert run.by_model[0].calls == 2
    assert run.cost_usd == pytest.approx(sum(m.cost_usd for m in run.by_model))


def test_meter_cache_hit_rate():
    m = cost.UsageMeter()
    # prompt tokens = input(100) + cache_read(300) = 400; hit rate = 300/400.
    m.record("claude-opus-4-8", _usage(i=100, o=10, cr=300))
    run = m.finalize()
    assert run.cache_hit_rate == pytest.approx(0.75)


def test_meter_ignores_missing_usage():
    m = cost.UsageMeter()
    m.record("claude-opus-4-8", None)  # fake clients return no usage
    run = m.finalize()
    assert run.calls == 0
    assert run.cost_usd == 0.0
    assert run.cache_hit_rate == 0.0


def test_meter_record_is_thread_safe():
    # Concurrent record() must not lose updates (the judge/edit loops are parallel).
    m = cost.UsageMeter()

    def worker():
        for _ in range(200):
            m.record("claude-opus-4-8", _usage(i=1, o=1))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    run = m.finalize()
    assert run.calls == 8 * 200
    assert run.input_tokens == 8 * 200
    assert run.output_tokens == 8 * 200


def test_meter_keys_by_model_and_stage():
    # judge and critique are both Haiku — stage keying keeps them separable.
    m = cost.UsageMeter()
    m.record("claude-haiku-4-5", _usage(i=10, o=2), stage="judge")
    m.record("claude-haiku-4-5", _usage(i=20, o=4), stage="critique")
    m.record("claude-haiku-4-5", _usage(i=5, o=1), stage="judge")
    run = m.finalize()

    by_stage = {mu.stage: mu for mu in run.by_model}
    assert set(by_stage) == {"judge", "critique"}
    assert by_stage["judge"].calls == 2
    assert by_stage["critique"].calls == 1


# --- metering proxy ---------------------------------------------------------


class _FakeMessages:
    def __init__(self, usage):
        self._usage = usage
        self.seen_model = None

    def parse(self, *, model=None, output_format=None, **_kw):
        self.seen_model = model
        return SimpleNamespace(parsed_output=PageEdit(), usage=self._usage)


class _FakeClient:
    def __init__(self, usage):
        self.messages = _FakeMessages(usage)
        self.other_attr = "passthrough"


def test_metered_client_records_and_passes_through():
    meter = cost.UsageMeter()
    inner = _FakeClient(_usage(i=42, o=7))
    client = cost.MeteredClient(inner, meter)

    resp = client.messages.parse(model="claude-opus-4-8", output_format=PageEdit)
    # The real response object is returned untouched.
    assert isinstance(resp.parsed_output, PageEdit)
    # ...and the model arg reached the inner client.
    assert inner.messages.seen_model == "claude-opus-4-8"
    # Usage was recorded against the right model.
    run = meter.finalize()
    assert run.input_tokens == 42 and run.output_tokens == 7
    assert run.by_model[0].model == "claude-opus-4-8"


def test_metered_client_passes_through_other_attributes():
    inner = _FakeClient(_usage())
    client = cost.MeteredClient(inner, cost.UsageMeter())
    assert client.other_attr == "passthrough"


def test_metered_client_attributes_current_stage():
    # The ContextVar set by `stage(...)` is read inside parse and recorded.
    meter = cost.UsageMeter()
    client = cost.MeteredClient(_FakeClient(_usage(i=5, o=1)), meter)
    with cost.stage("judge"):
        client.messages.parse(model="claude-haiku-4-5", output_format=PageEdit)
    run = meter.finalize()
    assert run.by_model[0].stage == "judge"


# --- rendering --------------------------------------------------------------


def test_render_empty_usage_is_empty():
    assert cost.render_usage_md(None) == []
    assert cost.render_usage_console(None) == ""
    assert cost.render_usage_md(cost.UsageMeter().finalize()) == []  # 0 calls


def test_render_usage_md_and_console_nonempty():
    m = cost.UsageMeter()
    m.record("claude-opus-4-8", _usage(i=1000, o=200, cr=500))
    run = m.finalize()

    md = cost.render_usage_md(run)
    assert any("Cost" in line for line in md)
    assert any("claude-opus-4-8" in line for line in md)

    console = cost.render_usage_console(run)
    assert "est." in console and "claude-opus-4-8×1" in console


def test_render_console_includes_stage_label():
    m = cost.UsageMeter()
    m.record("claude-haiku-4-5", _usage(i=10, o=2), stage="judge")
    console = cost.render_usage_console(m.finalize())
    assert "claude-haiku-4-5/judge" in console
