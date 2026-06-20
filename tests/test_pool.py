"""Unit tests for the bounded-parallelism helper."""

from docsync.pool import run_parallel


def test_run_parallel_preserves_order_serial():
    # max_workers=1 takes the serial path.
    assert run_parallel(lambda x: x * 2, [1, 2, 3], max_workers=1) == [2, 4, 6]


def test_run_parallel_preserves_order_parallel():
    # The pooled path still returns results in input order.
    assert run_parallel(lambda x: x * 2, [1, 2, 3, 4], max_workers=4) == [2, 4, 6, 8]


def test_run_parallel_empty_items():
    assert run_parallel(lambda x: x, [], max_workers=8) == []


def test_run_parallel_clamps_workers_to_item_count():
    # More workers than items is fine — clamped internally; result is still correct.
    assert run_parallel(str, [1, 2], max_workers=64) == ["1", "2"]
