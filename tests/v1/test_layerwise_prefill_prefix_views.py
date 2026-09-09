# SPDX-License-Identifier: Apache-2.0
"""Retained chunk tensor views are not rebuilt for every warm restore."""

# ruff: noqa: F811

# Standard
from dataclasses import replace
from typing import Any

# Third Party
from lmcache.v1.memory_management import TensorMemoryObj
import pytest

# First Party
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillSyncBackend

# Local
from tests.v1.test_layerwise_prefill_async import _backend as _async_backend
from tests.v1.test_layerwise_prefill_async import _execute
from tests.v1.test_layerwise_prefill_sync import (
    _request,
    _step,
    _view,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401


@pytest.fixture
def view_accesses(monkeypatch: pytest.MonkeyPatch) -> list:
    accesses = []
    original = TensorMemoryObj.tensor

    def counting(self: TensorMemoryObj) -> Any:
        accesses.append(self)
        return original.fget(self)

    monkeypatch.setattr(TensorMemoryObj, "tensor", property(counting))
    return accesses


def _aligned_request(start: int, end: int, generation: int) -> Any:
    """256-aligned continuation steps so no partial chunk is ever rebuilt."""
    return replace(
        _request(0, generation=generation),
        compute_start=start,
        compute_end=end,
        restore_end=start,
        token_ids=tuple(range(1024)),
        block_ids_by_bank=tuple(
            tuple(tuple(range(1, 13)) for _ in (0, 1)) for _ in (0, 1)
        ),
    )


def test_async_warm_steps_reuse_retained_views(
    page_runtime: Any, view_accesses: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = page_runtime.engine()
    backend = _async_backend(engine)
    connector = engine.gpu_connector
    prepared = []
    original_prepare = connector.prepare_layerwise_prefill_row

    def recording(*args: Any, **kwargs: Any) -> Any:
        prepared.append((kwargs["direction"], list(args[1]), len(args[2])))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(connector, "prepare_layerwise_prefill_row", recording)
    retained = None
    for step, (start, end) in enumerate(((0, 256), (256, 512), (512, 768)), 1):
        records_before = len(prepared)
        accesses_before = len(view_accesses)
        _execute(engine, backend, [_aligned_request(start, end, step)])
        records = prepared[records_before:]
        fresh = sum(count for direction, _, count in records if direction)
        # Only fresh store chunks acquire views (store ticket + commit); warm
        # restores hand the connector the retained manifest views verbatim.
        assert len(view_accesses) - accesses_before == 2 * fresh
        current = [tensors for direction, tensors, _ in records if not direction]
        assert len(current) == 101
        if step >= 2:
            assert all(current)
        if retained is not None:
            for previous, tensors in zip(retained, current, strict=True):
                assert len(tensors) == len(previous) + 1
                assert all(a is b for a, b in zip(previous, tensors, strict=False))
        retained = current


def test_sync_warm_restore_reuses_retained_views(
    page_runtime: Any,
    view_accesses: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = page_runtime.engine()
    backend = LayerwisePrefillSyncBackend(engine, _view())
    engine._layerwise_prefill_window_backend = backend
    transfers = []
    original = engine.gpu_connector.transfer_layerwise_prefill_row

    def captured(
        kv_layer: list,
        cpu_chunks: list,
        starts: list,
        ends: list,
        plan: Any,
        /,
        **kwargs: Any,
    ) -> Any:
        transfers.append((kwargs["direction"], list(cpu_chunks)))
        return original(kv_layer, cpu_chunks, starts, ends, plan, **kwargs)

    monkeypatch.setattr(
        engine.gpu_connector, "transfer_layerwise_prefill_row", captured
    )
    retained = None
    for step, (start, end) in enumerate(((0, 256), (256, 512), (512, 768)), 1):
        accesses_before = len(view_accesses)
        transfers_before = len(transfers)
        _step(engine, backend, [_aligned_request(start, end, step)])
        step_transfers = transfers[transfers_before:]
        loads = [chunks for direction, chunks in step_transfers if not direction]
        stores = [chunks for direction, chunks in step_transfers if direction]
        fresh = sum(len(chunks) for chunks in stores)
        assert len(view_accesses) - accesses_before == 2 * fresh
        # Step 1 restores nothing; warm steps restore the full retained prefix.
        assert len(loads) == (0 if step == 1 else 101)
        if step == 2:
            retained = loads
        elif step == 3:
            for previous, tensors in zip(retained, loads, strict=True):
                assert len(tensors) == len(previous) + 1
                assert all(a is b for a, b in zip(previous, tensors, strict=False))
