"""Cost & usage metering — wrap any LLM client, accumulate tokens, estimate $.

Every docsync LLM stage (impact judge, edit generation, self-critique) talks to
the model through `client.messages.parse(...)`, and every stage takes an
injectable `client=`. `MeteredClient` is a transparent proxy around such a client:
it records the `usage` off each response and leaves `.parsed_output` untouched, so
metering is added by wrapping the client *once* in `pipeline.run` — with zero
changes to the individual call sites.

Prices are a built-in estimate (USD per *million* tokens) and can drift from your
real Anthropic bill, so `RunUsage.estimated` is always True. Update `PRICING` when
rates change. Both response shapes are handled: the SDK's `usage` object (attribute
access) and the `claude-code` CLI envelope's `usage` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ModelUsage, RunUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD per *million* tokens, by token class."""

    input: float
    output: float
    cache_write: float  # 5-minute cache write (ephemeral)
    cache_read: float


# Built-in estimate, keyed by model-id prefix (longest match wins). Update as
# Anthropic pricing changes; `RunUsage.estimated` flags these as estimates.
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4": ModelPricing(15.0, 75.0, 18.75, 1.50),
    "claude-sonnet-4": ModelPricing(3.0, 15.0, 3.75, 0.30),
    "claude-haiku-4": ModelPricing(1.0, 5.0, 1.25, 0.10),
}
# Unknown model → assume opus-class, so cost is never silently under-reported.
_FALLBACK = ModelPricing(15.0, 75.0, 18.75, 1.50)


def pricing_for(model: str | None) -> ModelPricing:
    """Return the price table for `model` by longest matching id prefix."""
    name = model or ""
    best: tuple[int, ModelPricing] | None = None
    for prefix, price in PRICING.items():
        if name.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), price)
    return best[1] if best else _FALLBACK


def _get(usage: Any, key: str) -> int:
    """Read a token count off an SDK usage object or a plain dict; 0 if absent/None."""
    if usage is None:
        return 0
    val = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
    return int(val or 0)


def cost_of(model: str | None, usage: Any) -> float:
    """Estimated USD for a single response's `usage` under `model`'s pricing."""
    p = pricing_for(model)
    return (
        _get(usage, "input_tokens") * p.input
        + _get(usage, "output_tokens") * p.output
        + _get(usage, "cache_creation_input_tokens") * p.cache_write
        + _get(usage, "cache_read_input_tokens") * p.cache_read
    ) / 1_000_000


class UsageMeter:
    """Mutable accumulator of per-model token usage across a run."""

    def __init__(self) -> None:
        self._by_model: dict[str, ModelUsage] = {}

    def record(self, model: str | None, usage: Any) -> None:
        """Add one response's usage. No-op when usage is missing (e.g. fake clients)."""
        if usage is None:
            return
        name = model or "(unknown)"
        mu = self._by_model.get(name)
        if mu is None:
            mu = ModelUsage(model=name)
            self._by_model[name] = mu
        mu.calls += 1
        mu.input_tokens += _get(usage, "input_tokens")
        mu.output_tokens += _get(usage, "output_tokens")
        mu.cache_creation_input_tokens += _get(usage, "cache_creation_input_tokens")
        mu.cache_read_input_tokens += _get(usage, "cache_read_input_tokens")
        mu.cost_usd += cost_of(model, usage)

    def finalize(self) -> RunUsage:
        """Collapse the accumulator into a serializable `RunUsage` (cost-sorted)."""
        by_model = sorted(
            self._by_model.values(), key=lambda m: m.cost_usd, reverse=True
        )
        run = RunUsage(by_model=by_model, estimated=True)
        for m in by_model:
            run.calls += m.calls
            run.input_tokens += m.input_tokens
            run.output_tokens += m.output_tokens
            run.cache_creation_input_tokens += m.cache_creation_input_tokens
            run.cache_read_input_tokens += m.cache_read_input_tokens
            run.cost_usd += m.cost_usd
        prompt = run.prompt_tokens
        run.cache_hit_rate = (
            run.cache_read_input_tokens / prompt if prompt else 0.0
        )
        return run


# ---------------------------------------------------------------------------
# Metering proxy
# ---------------------------------------------------------------------------


class _MeteredMessages:
    def __init__(self, inner: Any, meter: UsageMeter) -> None:
        self._inner = inner
        self._meter = meter

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        resp = self._inner.parse(*args, **kwargs)
        self._meter.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp


class MeteredClient:
    """Drop-in proxy around any client exposing `.messages.parse`, recording usage.

    Pass-through for every other attribute, so it substitutes anywhere the real
    `anthropic.Anthropic()` (or `ClaudeCodeClient`) is used.
    """

    def __init__(self, inner: Any, meter: UsageMeter) -> None:
        self._inner = inner
        self.messages = _MeteredMessages(inner.messages, meter)

    def __getattr__(self, name: str) -> Any:  # only hit for attrs we don't define
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_usage_md(usage: RunUsage | None) -> list[str]:
    """Markdown lines summarizing run cost — empty when nothing was metered."""
    if usage is None or usage.calls == 0:
        return []
    lines = ["### Cost (estimated)"]
    lines.append(
        f"~${usage.cost_usd:.4f} · {usage.calls} LLM call(s) · "
        f"{usage.prompt_tokens:,} in / {usage.output_tokens:,} out tokens · "
        f"prompt-cache hit {usage.cache_hit_rate * 100:.0f}%"
    )
    for m in usage.by_model:
        lines.append(f"- `{m.model}` — {m.calls} call(s), ~${m.cost_usd:.4f}")
    lines.append(
        "_Estimated from a built-in price table; may differ from your actual bill._"
    )
    return lines


def render_usage_console(usage: RunUsage | None) -> str:
    """One-line cost summary for the terminal — empty when nothing was metered."""
    if usage is None or usage.calls == 0:
        return ""
    breakdown = ", ".join(f"{m.model}×{m.calls}" for m in usage.by_model)
    return (
        f"docsync: ~${usage.cost_usd:.4f} est. over {usage.calls} call(s), "
        f"prompt-cache hit {usage.cache_hit_rate * 100:.0f}% ({breakdown})"
    )
