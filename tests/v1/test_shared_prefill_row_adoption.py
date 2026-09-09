# SPDX-License-Identifier: Apache-2.0
"""Real row publication/adoption with CPU slabs and deterministic pin monitoring."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import TensorMemoryAllocator, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedCPUCacheValidationError,
)
import lmcache.v1.pin_monitor as pin_monitor_module
import pytest
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


@pytest.fixture(params=[0, 1])
def rows(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    monitor = PinMonitor(
        SimpleNamespace(pin_check_interval_sec=3600, pin_timeout_sec=3600)
    )
    monitor.stop_monitoring()
    monkeypatch.setattr(PinMonitor, "GetOrCreate", lambda config=None: monitor)
    wire = []
    engines = []
    for rank in (0, 1):
        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = LMCacheMetadata(
            "model", 2, 2, rank, rank, torch.bfloat16, (3, 1, 8, 1, 2), use_mla=True
        )
        engine.config = SimpleNamespace(dsa_two_groups=True, remote_url=None)
        engine.dsa_two_groups = True
        engine.shared_cpu_cache_name = "row-slab"
        engine.shared_cpu_cache_generation = 11
        engine.shared_cpu_cache_strict = True
        engine.gpu_connector = SimpleNamespace(
            get_shape=lambda tokens, kv_group: torch.Size(
                [tokens * (2 if kv_group == 0 else 1)]
            )
        )
        engines.append(engine)
    root, passive = engines
    root.broadcast_object_fn = lambda obj, src: wire.append(deepcopy(obj))
    passive.broadcast_object_fn = lambda obj, src: wire.pop(0)
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8), 64)
    root.storage_manager = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(
            memory_allocator=SimpleNamespace(
                shm_name="row-slab", pin_allocator=allocator, buffer=allocator.buffer
            )
        )
    )
    passive.shared_cpu_cache_passive_allocator = PassiveSharedViewAllocator(
        slab_tensor=allocator.buffer, shm_name="row-slab", generation=11
    )
    group = request.param
    starts = list(range(0, 17 * 8, 8))
    ends = [start + (3 if index == 16 else 8) for index, start in enumerate(starts)]
    sources = []
    keys = []
    with monitor.protect_pins() as pins:
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            shape, dtype, fmt = root.layerwise_prefill_row_metadata(group, end - start)
            obj = allocator.allocate(shape, dtype, fmt)
            assert obj is not None
            obj.tensor.fill_(index)
            obj.metadata.cached_positions = torch.arange(start, end)
            obj.pin()
            pins.append(obj)
            sources.append(obj)
            keys.append(
                CacheEngineKey("model", 2, 0, index, dtype, kv_group=group).get_layer(2)
            )
    arguments = dict(
        request_id="request",
        generation=7,
        group=group,
        row=2,
        keys=keys,
        starts=starts,
        ends=ends,
        phase="load",
    )
    assert (
        root.resolve_layerwise_prefill_row(**arguments, memory_objs=sources) is sources
    )
    created = []
    create_view = passive.shared_cpu_cache_passive_allocator.create_view

    def create(*args: Any, **kwargs: Any) -> TensorMemoryObj:
        obj = create_view(*args, **kwargs)
        created.append(obj)
        return obj

    views = Mock(side_effect=create)
    monkeypatch.setattr(
        passive.shared_cpu_cache_passive_allocator, "create_view", views
    )
    protect = Mock(wraps=monitor.protect_pins)
    monkeypatch.setattr(monitor, "protect_pins", protect)
    yield SimpleNamespace(
        root=root,
        passive=passive,
        monitor=monitor,
        wire=wire,
        arguments=arguments,
        sources=sources,
        created=created,
        views=views,
        protect=protect,
    )
    for obj in created + sources:
        if obj.is_valid():
            monitor.release_pin_lease(obj)
            obj.ref_count_down()
    assert allocator.num_active_allocations == 0
    assert monitor.get_monitored_count() == 0


def _resolve(rows: SimpleNamespace, **kwargs: Any) -> list[TensorMemoryObj]:
    return rows.passive.resolve_layerwise_prefill_row(**(rows.arguments | kwargs))


def _release(rows: SimpleNamespace, objects: list[TensorMemoryObj]) -> None:
    for obj in objects:
        rows.monitor.release_pin_lease(obj)
        obj.ref_count_down()
        assert (obj.get_ref_count(), obj.metadata.pin_count) == (0, 0)
        assert not obj.is_valid()
        with pytest.raises(ValueError, match="no protected pin lease"):
            rows.monitor.release_pin_lease(obj)


def test_one_adoption_transaction_per_row_preserves_views_and_ownership(
    rows: SimpleNamespace,
) -> None:
    first = _resolve(rows)
    rows.root.resolve_layerwise_prefill_row(**rows.arguments, memory_objs=rows.sources)
    second = _resolve(rows)
    assert rows.views.call_count == 34
    assert rows.protect.call_count == 2
    for source, a, b in zip(rows.sources, first, second, strict=True):
        assert a is not b
        assert a.tensor.data_ptr() == b.tensor.data_ptr() == source.tensor.data_ptr()
        assert torch.equal(a.tensor, source.tensor)
        assert torch.equal(
            a.metadata.cached_positions, source.metadata.cached_positions
        )
        for obj in (source, a, b):
            assert (obj.get_ref_count(), obj.metadata.pin_count) == (1, 1)
    _release(rows, first)
    assert all(obj.is_valid() and obj.metadata.pin_count == 1 for obj in second)
    _release(rows, second)
    assert all(obj.is_valid() and obj.metadata.pin_count == 1 for obj in rows.sources)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "wrong"),
        ("phase", "wrong"),
        ("layer_id", 3),
        ("kv_group", 9),
        ("chunk_index", 7),
        ("shm_name", "wrong"),
        ("generation", 12),
        ("producer_rank", 1),
        ("status", "miss"),
        ("key", None),
        ("shape", [3]),
        ("dtype", "float32"),
        ("fmt", 0),
        ("offset", -1),
        ("physical_size", 4096),
        ("logical_size", 1),
        ("cached_positions", None),
        ("shapes", [[2]]),
        ("dtypes", ["float32"]),
    ],
)
def test_later_handle_mutation_rejects_and_releases_only_adopted_prefix(
    rows: SimpleNamespace, field: str, value: Any
) -> None:
    handle = rows.wire[0]["handles"][8]
    if field == "key":
        handle[field] = replace(handle[field], chunk_hash=999)
    elif field == "cached_positions":
        # Same extent and length, wrong interior position must still fail.
        handle[field][3] += 1
    else:
        handle[field] = value
    with pytest.raises(
        SharedCPUCacheValidationError, match="Invalid shared CPU cache handle"
    ):
        _resolve(rows)
    assert rows.views.call_count == 9
    assert len(rows.created) == 8
    for obj in rows.created:
        assert (obj.get_ref_count(), obj.metadata.pin_count) == (0, 0)
        assert not obj.is_valid()
        with pytest.raises(ValueError, match="no protected pin lease"):
            rows.monitor.release_pin_lease(obj)
    assert all(
        (obj.get_ref_count(), obj.metadata.pin_count) == (1, 1) for obj in rows.sources
    )
    # A failed batch must not poison the next row or retain its views.
    rows.root.resolve_layerwise_prefill_row(**rows.arguments, memory_objs=rows.sources)
    _release(rows, _resolve(rows))


@pytest.mark.parametrize("field", ["starts", "ends"])
@pytest.mark.parametrize("extra", [False, True])
def test_strict_row_alignment_failure_cleans_partial_batch(
    rows: SimpleNamespace, field: str, extra: bool
) -> None:
    values = rows.arguments[field]
    values = values + [values[-1] + 8] if extra else values[:-1]
    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        _resolve(rows, **{field: values})
    assert len(rows.created) == (17 if extra else 16)
    assert all(
        not obj.is_valid() and obj.metadata.pin_count == 0 for obj in rows.created
    )


@pytest.mark.parametrize(
    "field,value",
    [("request_ordinal", 8), ("generation", 12), ("handles", [])],
)
def test_envelope_failure_precedes_any_adoption(
    rows: SimpleNamespace, field: str, value: Any
) -> None:
    rows.wire[0][field] = value
    with pytest.raises(ValueError):
        _resolve(rows)
    rows.views.assert_not_called()
    rows.protect.assert_not_called()


def test_skipped_empty_row_needs_no_pin_transaction(rows: SimpleNamespace) -> None:
    rows.wire.clear()
    arguments = rows.arguments | dict(keys=[], starts=[], ends=[])
    rows.root.resolve_layerwise_prefill_row(**arguments, memory_objs=[])
    assert _resolve(rows, keys=[], starts=[], ends=[]) == []
    rows.views.assert_not_called()
    rows.protect.assert_not_called()


def test_timeout_candidate_cannot_expire_a_pin_during_batch_adoption(
    rows: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [pin_monitor_module.time.time()]
    monkeypatch.setattr(
        pin_monitor_module, "time", SimpleNamespace(time=lambda: clock[0])
    )
    pinned, proceed, selected = Event(), Event(), Event()
    on_pin = rows.monitor.on_pin
    force = rows.monitor._force_unpin_timeout_object

    def pause_after_pin(obj: TensorMemoryObj) -> None:
        on_pin(obj)
        if not pinned.is_set():
            clock[0] += 1_000_000
            pinned.set()
            assert proceed.wait(5)

    def selected_timeout(*args: Any) -> bool:
        selected.set()
        return force(*args)

    monkeypatch.setattr(rows.monitor, "on_pin", pause_after_pin)
    monkeypatch.setattr(rows.monitor, "_force_unpin_timeout_object", selected_timeout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        adopted = pool.submit(_resolve, rows)
        assert pinned.wait(5)
        swept = pool.submit(rows.monitor._check_timeouts)
        try:
            assert selected.wait(5)
            assert not swept.done()
        finally:
            proceed.set()
        objects = adopted.result(timeout=5)
        assert swept.result(timeout=5)[2] == 0
    clock[0] += 1_000_000
    assert rows.monitor._check_timeouts()[2] == 0
    assert all(obj.metadata.pin_count == 1 for obj in objects)
    _release(rows, objects)
