"""Thin wrapper over the Anthropic structured-output call.

Every stage (impact judge, edit, critique, polish, bootstrap plan/author) makes the
same ``client.messages.parse(...)`` call: a cached system prefix, one user message,
a structured ``output_format``, all wrapped in a ``cost.stage(...)`` meter. The only
differences are the model, token budget, whether the call thinks (Opus) and which
output model it returns. This collapses that shared shape into one place so the
cache-control block and the metering live in exactly one spot.
"""

from __future__ import annotations

from . import cost


def get_client(client=None):
    """Return ``client`` or lazily build a default ``anthropic.Anthropic()``.

    The stages accept an injected client (real runs + tests pass one); when absent
    they fall back to a default client so the module is usable standalone.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    return client


def parse(
    client,
    *,
    stage: str,
    model: str,
    max_tokens: int,
    system: str | list[dict],
    user: str,
    output_format,
    cache_system: bool = True,
    thinking: bool = False,
    effort: str | None = None,
):
    """Run one metered ``messages.parse`` and return its ``.parsed_output``.

    ``system`` may be a plain string (wrapped in a single text block, cached when
    ``cache_system``) or a pre-built list of system blocks passed through unchanged —
    used by the edit stage, which appends a second cached block for the shared diff.
    Opus callers set ``thinking=True`` and an ``effort``; Haiku judges omit both.
    """
    if isinstance(system, str):
        block: dict = {"type": "text", "text": system}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        system_blocks: list[dict] = [block]
    else:
        system_blocks = system

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user}],
        "output_format": output_format,
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}

    with cost.stage(stage):
        resp = client.messages.parse(**kwargs)
    return resp.parsed_output
