"""Bounded parallelism helper.

Every stage that fans out over independent items (judge candidates, edit pages,
author pages, infer pages) repeats the same shape: clamp the worker count to
``config.max_parallel_requests``, run serially when one worker would do, else map
over a thread pool. This collapses that into one place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], items: Sequence[T], max_workers: int) -> list[R]:
    """Map ``fn`` over ``items`` with at most ``max_workers`` threads, preserving order.

    Runs serially (no pool) when a single worker would suffice — cheaper and keeps
    stack traces flat for the common one-item case. ``ThreadPoolExecutor.map`` preserves
    input order, so the returned list lines up with ``items``.
    """
    workers = max(1, min(max_workers, len(items)))
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, items))
