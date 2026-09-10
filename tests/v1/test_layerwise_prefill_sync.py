# SPDX-License-Identifier: Apache-2.0
"""CPU sentinels exercise production P manifests, storage and TP handle exchange."""

# Standard
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import timedelta
from queue import Queue
from threading import Barrier, Event, Lock, local
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import gc
import multiprocessing
import weakref

# Third Party
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import ChunkedTokenDatabase
import lmcache.v1.config as lmcache_config
import lmcache.v1.gpu_connector.gpu_connectors as gpu_connectors_module
import lmcache.v1.pin_monitor as pin_monitor_module
import pytest
import torch
import vllm.distributed.parallel_state

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.dsa_kv_topology import DSAKVTopologyView
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillFenceError,
    LayerwisePrefillSyncBackend,
)
from lmcache_ascend.v1.layerwise_prefill_window import LayerwisePrefillNPUWindowBackend
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)
from lmcache_ascend.v1.npu_connector.utils import permute_kv_caches_to_contiguous
import lmcache_ascend.v1.layerwise_prefill_sync as sync_module

INDEXER_EXECUTIONS = (0, 1, 2, *range(6, 79, 4))


def _view() -> DSAKVTopologyView:
    groups = tuple(
        tuple(
            (f"g{group}.e{execution}", execution, group, row, row % 2)
            for row, execution in enumerate(executions)
        )
        for group, executions in enumerate((range(79), INDEXER_EXECUTIONS))
    )
    indexers = {row[1]: row for row in groups[1]}
    return DSAKVTopologyView(
        "sync-79-22",
        groups,
        tuple((index, row, indexers.get(index)) for index, row in enumerate(groups[0])),
    )


def _metadata(view: DSAKVTopologyView, key: tuple, requests: list) -> Any:
    def row(value: tuple | None) -> Any:
        return (
            None
            if value is None
            else SimpleNamespace(
                **dict(
                    zip(
                        (
                            "layer_name",
                            "execution_ordinal",
                            "kv_group",
                            "row_ordinal",
                            "bank",
                        ),
                        value,
                        strict=True,
                    )
                )
            )
        )

    execution = view.executions[key[1]]
    return SimpleNamespace(
        row=row(key),
        request_generations=tuple(
            (req.request_id, req.allocation_generation) for req in requests
        ),
        execution=SimpleNamespace(
            execution_ordinal=execution[0],
            latent=row(execution[1]),
            indexer=row(execution[2]),
        ),
    )


def _request(
    index: int = 0, start: int = 0, end: int = 300, **kwargs: Any
) -> LayerwisePrefillRequest:
    return LayerwisePrefillRequest(
        request_id=kwargs.pop("request_id", f"req-{index}"),
        allocation_generation=kwargs.pop("generation", 1),
        token_ids=tuple(range(index * 1000, index * 1000 + 600)),
        compute_start=start,
        compute_end=end,
        restore_end=kwargs.pop("restore_end", start),
        block_ids_by_bank=tuple(
            tuple(
                tuple(range(1 + index * 12 + bank * 6, 6 + index * 12 + bank * 6))
                for _ in (0, 1)
            )
            for bank in (0, 1)
        ),
        block_size=128,
        request_configs={"lmcache.tag.tenant": "sentinel"},
        **kwargs,
    )


def _slots(
    req: LayerwisePrefillRequest, bank: int, group: int, end: int
) -> torch.Tensor:
    pos = torch.arange(end)
    blocks = torch.tensor(req.block_ids_by_bank[bank][group])
    return blocks[pos // req.block_size] * req.block_size + pos % req.block_size


def _sentinel(
    req: LayerwisePrefillRequest, group: int, row: int, plane: int, end: int
) -> torch.Tensor:
    return (
        torch.arange(end) % 7
        + req.token_ids[0] // 1000 * 3
        + group * 32
        + row
        + plane * 17
    ).to(torch.bfloat16)


@dataclass(eq=False)
class _CPUSlotPlan:
    snapshot: torch.Tensor
    group: int
    capacity: int


class CPUConnector:
    """Exact packed-plane transfer, with all rows aliasing two recyclable banks."""

    def __init__(self, blocks: int = 64) -> None:
        self.layouts = {}
        self.calls = []
        self.prepared = []
        self.plan_uses = Counter()
        self.fail_load = False
        self.fail_store = False
        self.planes = {
            group: [
                torch.full((blocks, 128, 1, 1), -100, dtype=torch.bfloat16)
                for _ in range(2 if group == 0 else 1)
            ]
            for group in (0, 1)
        }

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        # Exercise the production entry-container ABI, not a permissive copy.
        GPUConnectorInterface.initialize_kvcaches_ptr(self, **kwargs)

    def _lazy_initialize_buffer(
        self, caches: list, *, kv_group: int, init_staging: bool
    ) -> None:
        assert init_staging is False
        assert len(caches) == (79 if kv_group == 0 else 22)
        self.layouts[kv_group] = len(caches)

    def get_num_layers(self, group: int) -> int | None:
        return self.layouts.get(group)

    def get_shape(self, tokens: int, kv_group: int) -> torch.Size:
        assert self.layouts == {0: 79, 1: 22}
        return torch.Size([tokens * (2 if kv_group == 0 else 1)])

    def synchronize_dense_load_stream(self) -> None:
        pass

    def synchronize_shared_cpu_store_publication(self) -> None:
        pass

    def prepare_layerwise_prefill_slots(
        self, mapping: torch.Tensor, *, kv_group: int, capacity: int
    ) -> _CPUSlotPlan:
        assert [self.get_num_layers(group) for group in (0, 1)] == [79, 22]
        assert mapping.device.type == "cpu" and mapping.dtype == torch.long
        assert mapping.ndim == 1 and mapping.is_contiguous()
        plane = self.planes[kv_group][0]
        assert capacity == plane.shape[0] * plane.shape[1]
        assert bool(((mapping >= 0) & (mapping < capacity)).all())
        plan = _CPUSlotPlan(mapping.clone(), kv_group, capacity)
        self.prepared.append(weakref.ref(plan))
        return plan

    def transfer_layerwise_prefill_row(
        self,
        kv_layer: list,
        cpu_chunks: list,
        starts: list,
        ends: list,
        plan: _CPUSlotPlan,
        /,
        *,
        kv_group: int,
        direction: bool,
    ) -> None:
        # The backend must pass the exact opaque return value positionally.
        prepared = [
            i for i, reference in enumerate(self.prepared) if reference() is plan
        ]
        assert len(prepared) == 1
        assert plan.group == kv_group
        self.plan_uses[prepared[0], direction] += 1
        self._transfer_cpu_row(
            kv_layer,
            cpu_chunks,
            starts,
            ends,
            plan.snapshot,
            kv_group=kv_group,
            direction=direction,
        )

    def _transfer_cpu_row(
        self,
        kv_layer: list,
        cpu_chunks: list,
        starts: list,
        ends: list,
        slot_mapping: torch.Tensor,
        *,
        kv_group: int,
        direction: bool,
        slot_mapping_base: int = 0,
    ) -> None:
        assert self.layouts == {0: 79, 1: 22}
        assert slot_mapping_base == 0
        self.calls.append((direction, kv_group, tuple(starts), tuple(ends)))
        if (direction and self.fail_store) or (not direction and self.fail_load):
            raise RuntimeError("CPU transfer sentinel failure")
        for chunk, start, end in zip(cpu_chunks, starts, ends, strict=True):
            assert isinstance(chunk, torch.Tensor)
            packed = chunk.view(len(kv_layer), end - start)
            slots = slot_mapping[start:end]
            for index, plane in enumerate(kv_layer):
                if direction:
                    packed[index].copy_(plane.view(-1)[slots])
                else:
                    plane.view(-1)[slots] = packed[index]


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    # --noconftest CPU runs do not install every Ascend monkeypatch. Match the
    # production GPUConnectorInterface -> NPU permutation dispatch explicitly.
    monkeypatch.setattr(
        gpu_connectors_module,
        "permute_kv_caches_to_contiguous",
        permute_kv_caches_to_contiguous,
    )
    thread = local()
    serving = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )
    channel = Queue()
    barrier = Barrier(2, timeout=10)
    statuses = [None, None]
    world = SimpleNamespace(size=1)
    engines = []
    real_gather = torch.distributed.all_gather_object
    real_tensor_gather = torch.distributed.all_gather
    real_initialized = torch.distributed.is_initialized

    def tp() -> Any:
        return SimpleNamespace(
            world_size=world.size,
            rank_in_group=getattr(thread, "rank", 0),
            cpu_group=barrier,
        )

    def gather(out: list, status: Any, group: Any) -> None:
        tensor = isinstance(status, torch.Tensor)
        if group is not barrier:
            return (real_tensor_gather if tensor else real_gather)(
                out, status, group=group
            )
        statuses[thread.rank] = status.clone() if tensor else deepcopy(status)
        barrier.wait()
        if tensor:
            for target, peer in zip(out, statuses, strict=True):
                target.copy_(peer)
        else:
            out[:] = statuses
        barrier.wait()

    monkeypatch.setattr(vllm.distributed.parallel_state, "get_tp_group", tp)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(torch.distributed, "all_gather", gather)

    def engine(
        rank: int = 0,
        root: Any = None,
        size: int = 1,
        *,
        blocks: int = 64,
        slab_bytes: int = 8 << 20,
    ) -> Any:
        world.size = serving.parallel_config.tensor_parallel_size = size
        thread.rank = rank
        config = lmcache_config.LMCacheEngineConfig.from_defaults()
        for key, value in dict(
            use_layerwise=True,
            dsa_two_groups=True,
            local_cpu=True,
            enable_shared_cpu_cache=True,
            shared_cpu_cache_strict=True,
            store_async=False,
            save_unfull_chunk=True,
            chunk_size=256,
            max_local_cpu_size=1,
            extra_config={},
        ).items():
            setattr(config, key, value)
        metadata = LMCacheMetadata(
            "sync-model",
            size,
            size,
            rank,
            rank,
            torch.bfloat16,
            (79, 1, 256, 1, 2),
            use_mla=True,
        )
        PinMonitor.GetOrCreate(config)
        result = object.__new__(AscendLMCacheEngine)
        result.config, result.metadata = config, metadata
        result.configure_layerwise_prefill_sync(serving)
        result.dsa_two_groups = result.use_layerwise = True
        result.save_only_first_rank = result.save_indexer_only_first_rank = True
        result.enable_shared_cpu_cache = result.shared_cpu_cache_strict = True
        result.shared_cpu_cache_name = "sync-test-slab"
        result.shared_cpu_cache_generation = 11
        result._dsa_kv_topology_view = _view()
        result.gpu_connector = CPUConnector(blocks)
        result.token_database = ChunkedTokenDatabase(config, metadata)

        def broadcast(obj: Any, src: int) -> Any:
            assert src == 0
            if rank == 0:
                channel.put(deepcopy(obj))
                return obj
            return channel.get(timeout=10)

        result.broadcast_object_fn = broadcast
        if rank == 0:
            allocator = TensorMemoryAllocator(
                torch.empty(slab_bytes, dtype=torch.uint8)
            )
            allocator.shm_name = result.shared_cpu_cache_name
            allocator.pin_allocator = allocator
            cpu = LocalCPUBackend(config, metadata, memory_allocator=allocator)
            manager = object.__new__(StorageManager)
            manager.allocator_backend = manager.local_cpu_backend = cpu
            manager.storage_backends = {"LocalCPUBackend": cpu}
            manager._bypass_lock = Lock()
            manager._bypassed_backends = set()
            manager._freeze_lock = Lock()
            manager._freeze = False
            result.storage_manager = manager
        else:
            result.storage_manager = None
            result.shared_cpu_cache_passive_allocator = PassiveSharedViewAllocator(
                slab_tensor=root.storage_manager.local_cpu_backend.memory_allocator.buffer,
                shm_name=result.shared_cpu_cache_name,
                generation=11,
            )
        engines.append(result)
        return result

    def parallel(*functions: Any) -> list:
        def run(rank: int, function: Any) -> Any:
            thread.rank = rank
            return function()

        with ThreadPoolExecutor(max_workers=len(functions)) as pool:
            futures = [
                pool.submit(run, rank, function)
                for rank, function in enumerate(functions)
            ]
            return [future.result(timeout=20) for future in futures]

    yield SimpleNamespace(
        engine=engine,
        parallel=parallel,
        serving=serving,
        thread=thread,
        real_gather=real_gather,
        real_initialized=real_initialized,
    )
    for result in reversed(engines):
        backend = getattr(result, "_layerwise_prefill_window_backend", None)
        if backend is not None:
            for request_id in ("seed", "external", *(f"req-{i}" for i in range(4))):
                backend.abort_request(request_id)
        if result.storage_manager is not None:
            cpu = result.storage_manager.local_cpu_backend
            for key in list(cpu.hot_cache):
                cpu.remove(key, force=True)
            assert cpu.memory_allocator.num_active_allocations == 0


def _registry(engine: Any) -> dict:
    return {
        row[0]: engine.gpu_connector.planes[group]
        for group, rows in enumerate(_view().rows_by_group)
        for row in rows
    }


def _patch_cpu_slot_preparer(
    monkeypatch: pytest.MonkeyPatch, connector: Any, registry: dict
) -> CPUConnector:
    """Keep real layout initialization, replacing only the NPU-only preparation."""
    sentinel = CPUConnector()
    sentinel.planes = {
        group: registry[rows[0][0]] for group, rows in enumerate(_view().rows_by_group)
    }
    monkeypatch.setattr(sentinel, "get_num_layers", connector.get_num_layers)
    monkeypatch.setattr(
        connector,
        "prepare_layerwise_prefill_slots",
        sentinel.prepare_layerwise_prefill_slots,
    )
    return sentinel


@pytest.mark.parametrize("container", [list, tuple])
def test_bind_initializes_real_connector_without_copying_kv_planes(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, container: Any
) -> None:
    engine = runtime.engine()
    registry = {name: container(planes) for name, planes in _registry(engine).items()}
    connector = VLLMPagedMemLayerwiseNPUConnector.__new__(
        VLLMPagedMemLayerwiseNPUConnector
    )
    connector.use_mla = connector.dsa_two_groups = True
    connector.use_gpu = False
    connector.lmcache_chunk_size = 256
    connector._group_layouts = {}
    connector._dsa_kv_topology_view = _view()
    engine.gpu_connector = connector
    normalized_groups = []

    def normalize(caches: list) -> list:
        assert all(isinstance(planes, tuple) for planes in caches)
        result = permute_kv_caches_to_contiguous(caches)
        normalized_groups.append(result)
        for before, after in zip(caches, result, strict=True):
            for src, dst in zip(before, after, strict=True):
                assert dst.data_ptr() == src.data_ptr()
                assert dst.stride() == src.stride()
        return result

    # Keep the actual GPUConnectorInterface initializer, Ascend permutation,
    # format detection and per-group lazy initialization in the exercised path.
    monkeypatch.setattr(
        gpu_connectors_module, "permute_kv_caches_to_contiguous", normalize
    )
    sentinel = _patch_cpu_slot_preparer(monkeypatch, connector, registry)
    backend = engine.layerwise_prefill_window_backend
    backend.bind_step([_request()], registry)
    assert len(sentinel.prepared) == 4
    assert all(isinstance(plan, _CPUSlotPlan) for plan in backend._slots.values())
    assert [len(group) for group in normalized_groups] == [79, 22]
    assert [connector.get_num_layers(group) for group in (0, 1)] == [79, 22]
    assert [connector.get_shape(17, kv_group=group).numel() for group in (0, 1)] == [
        34,
        17,
    ]
    assert all(
        layout.gpu_buffer_allocator is None
        for layout in connector._group_layouts.values()
    )
    for name, planes in registry.items():
        assert isinstance(planes, container)
        assert isinstance(backend._caches[name], tuple)
        assert all(
            src is dst for src, dst in zip(planes, backend._caches[name], strict=True)
        )


def _step(engine: Any, backend: Any, requests: list) -> None:
    registry = _registry(engine)
    view = _view()
    backend.bind_step(requests, registry)
    for _, latent, indexer in view.executions:
        for key in (latent, indexer):
            if key is None:
                continue
            group, row, bank = key[2:]
            metadata = _metadata(view, key, requests)
            backend.wait_for_load(metadata)
            for req in requests:
                slots = _slots(req, bank, group, req.compute_end)
                for index, plane in enumerate(registry[key[0]]):
                    expected = _sentinel(req, group, row, index, req.compute_end)
                    assert torch.equal(
                        plane.view(-1)[slots[: req.restore_end]],
                        expected[: req.restore_end],
                    )
                    plane.view(-1)[slots[req.compute_start :]] = expected[
                        req.compute_start :
                    ]
            backend.sync_save(metadata, registry[key[0]])
            for plane in registry[key[0]]:
                plane.fill_(-100)
    backend.finish_step()


def _count_manifest_sources(
    monkeypatch: pytest.MonkeyPatch, engine: Any, backend: Any
) -> Counter:
    counts = Counter()

    def track(target: Any, method_name: str, counter: str) -> None:
        method = getattr(target, method_name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[counter] += len(kwargs["keys_layer"]) if counter == "handles" else 1
            return method(*args, **kwargs)

        monkeypatch.setattr(target, method_name, counted)

    track(backend, "_plan", "plans")
    track(engine, "resolve_layerwise_prefill_row", "resolves")
    track(engine, "_make_shared_handles_for_layer", "handles")
    track(engine, "broadcast_object_fn", "broadcasts")
    track(engine, "layerwise_prefill_ack", "acks")
    if not engine.metadata.is_first_rank():
        track(engine.shared_cpu_cache_passive_allocator, "create_view", "views")
    return counts


@pytest.mark.parametrize("request_count", [1, 4])
def test_bind_builds_common_cpu_slot_arithmetic_once_per_request(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, request_count: int
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    requests = [_request(i) for i in range(request_count)]
    positions = Mock(wraps=torch.arange)
    arithmetic = Counter()
    divide, modulo = torch.Tensor.__floordiv__, torch.Tensor.__mod__

    def block_indices(tensor: torch.Tensor, size: int) -> torch.Tensor:
        arithmetic["divide"] += 1
        assert tensor.device.type == "cpu"
        return divide(tensor, size)

    def block_offsets(tensor: torch.Tensor, size: int) -> torch.Tensor:
        arithmetic["modulo"] += 1
        assert tensor.device.type == "cpu"
        return modulo(tensor, size)

    with monkeypatch.context() as bind:
        bind.setattr(torch, "arange", positions)
        bind.setattr(torch.Tensor, "__floordiv__", block_indices)
        bind.setattr(torch.Tensor, "__mod__", block_offsets)
        backend.bind_step(requests, _registry(engine))
    assert positions.call_count == request_count
    assert arithmetic == Counter(divide=request_count, modulo=request_count)
    for call, req in zip(positions.call_args_list, requests, strict=True):
        assert call.args == (req.compute_end,)
        assert call.kwargs == {"dtype": torch.long, "device": "cpu"}
    assert len(engine.gpu_connector.prepared) == 4 * request_count
    assert len(backend._slots) == 4 * request_count
    for req in requests:
        for bank in (0, 1):
            for group in (0, 1):
                plan = backend._slots[req.request_id, bank, group]
                assert plan.group == group and plan.capacity == 64 * 128
                assert torch.equal(
                    plan.snapshot, _slots(req, bank, group, req.compute_end)
                )
    assert not engine.gpu_connector.calls


@pytest.mark.parametrize("prepare_hook", ["absent", "noncallable"])
def test_connector_without_preparer_keeps_cpu_tensor_transfer_contract(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, prepare_hook: str
) -> None:
    engine = runtime.engine()
    connector = engine.gpu_connector
    if prepare_hook == "absent":
        monkeypatch.delattr(CPUConnector, "prepare_layerwise_prefill_slots")
    else:
        monkeypatch.setattr(connector, "prepare_layerwise_prefill_slots", None)
    monkeypatch.setattr(
        connector, "transfer_layerwise_prefill_row", connector._transfer_cpu_row
    )
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request()])
    _step(engine, backend, [_request(start=300, end=530)])
    assert not connector.prepared and not connector.plan_uses
    assert backend._prepared_slot_count == 0
    assert Counter(call[0] for call in connector.calls) == {True: 202, False: 101}


@pytest.mark.parametrize("request_count", [1, 4])
@pytest.mark.parametrize("save_row", [False, True])
def test_abort_releases_bound_slot_plans(
    runtime: Any, request_count: int, save_row: bool
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    requests = [_request(i) for i in range(request_count)]
    registry = _registry(engine)
    backend.bind_step(requests, registry)
    refs = list(engine.gpu_connector.prepared)
    assert len(refs) == 4 * request_count and all(ref() is not None for ref in refs)
    if save_row:
        metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
        backend.wait_for_load(metadata)
        backend.sync_save(metadata, registry[metadata.row.layer_name])
    for req in requests:
        backend.abort_request(req.request_id)
    assert not backend._slots and not backend._plans and not backend._requests
    assert all(ref() is None for ref in refs)
    assert not backend._bound and not backend._prefixes


@pytest.mark.parametrize("bad_rank", [0, 1])
@pytest.mark.parametrize("fail_at", [0, 3])
def test_slot_preparation_failure_reaches_all_tp_before_row_transfer(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, bad_rank: int, fail_at: int
) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    engines = (root, passive)
    backends = []
    for rank, engine in enumerate(engines):
        runtime.thread.rank = rank
        backends.append(engine.layerwise_prefill_window_backend)
    connector = engines[bad_rank].gpu_connector
    prepare = connector.prepare_layerwise_prefill_slots

    def fail(mapping: torch.Tensor, *, kv_group: int, capacity: int) -> Any:
        if len(connector.prepared) == fail_at:
            raise RuntimeError("slot preparation sentinel failure")
        return prepare(mapping, kv_group=kv_group, capacity=capacity)

    monkeypatch.setattr(connector, "prepare_layerwise_prefill_slots", fail)
    counts = [
        _count_manifest_sources(monkeypatch, engine, backend)
        for engine, backend in zip(engines, backends, strict=True)
    ]

    def bind(rank: int) -> None:
        backend = backends[rank]
        with pytest.raises(ValueError, match="slot preparation sentinel failure"):
            backend.bind_step([_request()], _registry(engines[rank]))
        assert not backend._bound and not backend._slots and not backend._prefixes

    runtime.parallel(lambda: bind(0), lambda: bind(1))
    assert counts == [Counter(acks=1), Counter(acks=1)]
    # Exception tracebacks can temporarily own the rejected bind's local plans.
    gc.collect()
    for rank, engine in enumerate(engines):
        assert not engine.gpu_connector.calls
        refs = engine.gpu_connector.prepared
        assert len(refs) == (fail_at if rank == bad_rank else 4)
        assert all(ref() is None for ref in refs)
    assert not root.storage_manager.local_cpu_backend.hot_cache


@pytest.mark.parametrize("request_count", [1, 4])
def test_bank_reuse_full_prefix_partial_successor_and_refcounts(
    runtime: Any, request_count: int
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    assert isinstance(backend, LayerwisePrefillSyncBackend)
    assert backend.supports_sync_callbacks and not backend.supports_transfer_window
    assert backend.persists_indexer_group and backend.layer_count(1) == 22
    requests = [_request(index) for index in range(request_count)]
    _step(engine, backend, requests)
    cpu = engine.storage_manager.local_cpu_backend
    old = list(cpu.hot_cache.values())
    assert all(obj.get_ref_count() == 2 and obj.metadata.pin_count == 1 for obj in old)
    # Evict the hot-cache references forcibly: continuations must use their
    # own manifests, not repeat a storage hit lookup.
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)
    successors = [
        replace(
            req,
            compute_start=300,
            compute_end=530,
            restore_end=300,
            allocation_generation=2,
        )
        for req in requests
    ]
    _step(engine, backend, successors)
    assert sum(obj.is_valid() for obj in old) == request_count * 101
    calls = engine.gpu_connector.calls[request_count * 101 :]
    assert any(call[2:] == ((256, 512), (512, 530)) for call in calls if call[0])
    for req in successors:
        backend.abort_request(req.request_id)
    assert all(
        obj.metadata.pin_count == 0 and obj.get_ref_count() == 1
        for obj in cpu.hot_cache.values()
    )
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)
    assert cpu.memory_allocator.num_active_allocations == 0


@pytest.mark.parametrize("start,end", [(300, 530), (299, 300)])
def test_external_partial_hit_uses_original_extent(
    runtime: Any, start: int, end: int
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    target = _request(start=start, end=end, restore_end=300, request_id="external")
    _step(engine, backend, [target])
    assert any(
        not direction and starts == (0, 256) and ends == (256, 300)
        for direction, _, starts, ends in engine.gpu_connector.calls
    )
    backend.abort_request("external")


@pytest.mark.parametrize("group", [0, 1])
def test_missing_required_source_fails_closed(runtime: Any, group: int) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    requests = [_request(start=300, end=530)]
    backend.bind_step(requests, _registry(engine))
    with pytest.raises(ValueError, match="Missing required"):
        backend.wait_for_load(
            _metadata(_view(), _view().rows_by_group[group][0], requests)
        )
    assert not engine.gpu_connector.calls
    with pytest.raises(ValueError, match="Incomplete"):
        backend.finish_step()
    backend.abort_request("req-0")


@pytest.mark.parametrize(
    "mutate", ["generation", "requests", "row", "execution", "planes", "save_first"]
)
def test_callback_exact_bindings(runtime: Any, mutate: str) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    requests = [_request()]
    registry = _registry(engine)
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    if mutate == "generation":
        metadata.request_generations = (("req-0", 2),)
    elif mutate == "requests":
        metadata.request_generations = (("other", 1),)
    elif mutate == "row":
        metadata.row.bank = 1
    elif mutate == "execution":
        metadata.execution.indexer.row_ordinal = 1
    if mutate == "planes":
        backend.wait_for_load(metadata)
    with pytest.raises(ValueError):
        if mutate in ("planes", "save_first"):
            backend.sync_save(
                metadata,
                [tensor.clone() for tensor in registry[metadata.row.layer_name]],
            )
        else:
            backend.wait_for_load(metadata)
    assert not engine.gpu_connector.calls
    backend.abort_request("req-0")


class RequiredStore:
    def __init__(self, cpu: Any, future: Future) -> None:
        self.cpu, self.future = cpu, future
        self.submitted = Event()
        self.connection = SimpleNamespace(support_batched_put=lambda: True)
        self.completion = True
        self.put_result = [future]
        self.calls = []

    def get_allocator_backend(self) -> Any:
        return self.cpu

    def requires_put_completion(self) -> bool:
        return self.completion

    def batched_submit_put_task(self, keys: list, objects: list, **kwargs: Any) -> list:
        self.calls.append((tuple(keys), tuple(id(obj) for obj in objects)))
        self.submitted.set()
        return self.put_result


@pytest.mark.parametrize("failure", [False, True])
def test_sync_save_waits_required_future(runtime: Any, failure: bool) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    future = Future()
    remote = RequiredStore(engine.storage_manager.local_cpu_backend, future)
    engine.storage_manager.storage_backends["RequiredStore"] = remote
    requests = [_request()]
    registry = _registry(engine)
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    backend.wait_for_load(metadata)
    with ThreadPoolExecutor(max_workers=1) as pool:
        saved = pool.submit(
            backend.sync_save, metadata, registry[metadata.row.layer_name]
        )
        assert remote.submitted.wait(5)
        assert not saved.done()
        if failure:
            future.set_exception(RuntimeError("persist failed"))
            with pytest.raises(ValueError, match="persist failed"):
                saved.result(timeout=5)
        else:
            future.set_result(None)
            saved.result(timeout=5)
    backend.abort_request("req-0")
    assert all(
        obj.metadata.pin_count == 0
        for obj in engine.storage_manager.local_cpu_backend.hot_cache.values()
    )


@pytest.mark.parametrize("request_count", [1, 4])
def test_tp_passive_restores_every_row_but_never_stores(
    runtime: Any, request_count: int
) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    runtime.thread.rank = 0
    root_backend = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    passive_backend = passive.layerwise_prefill_window_backend
    for step, (start, end) in enumerate(((0, 300), (300, 530))):
        requests = [
            _request(index, start, end, generation=step + 1)
            for index in range(request_count)
        ]
        runtime.parallel(
            lambda requests=requests: _step(root, root_backend, requests),
            lambda requests=requests: _step(passive, passive_backend, requests),
        )
        for engine, backend in ((root, root_backend), (passive, passive_backend)):
            connector = engine.gpu_connector
            assert len(connector.prepared) == (step + 1) * request_count * 4
            assert not backend._slots
            assert all(ref() is None for ref in connector.prepared)
            rows_per_bank = Counter(
                (row[4], group)
                for group, rows in enumerate(_view().rows_by_group)
                for row in rows
            )
            for index in range(step * request_count * 4, len(connector.prepared)):
                bank, group = divmod(index % 4, 2)
                rows = rows_per_bank[bank, group]
                assert connector.plan_uses[index, False] == (rows if step else 0)
                assert connector.plan_uses[index, True] == (
                    rows if engine is root else 0
                )
    assert len(passive.gpu_connector.calls) == request_count * 101
    assert all(not call[0] for call in passive.gpu_connector.calls)
    assert sum(call[0] for call in root.gpu_connector.calls) == 2 * request_count * 101
    for index in range(request_count):
        root_backend.abort_request(f"req-{index}")
        passive_backend.abort_request(f"req-{index}")
    assert all(
        obj.metadata.pin_count == 0
        for obj in root.storage_manager.local_cpu_backend.hot_cache.values()
    )


@pytest.mark.parametrize("capability", [None, False, 1, "true", True])
def test_step_progress_is_root_only_and_resets_timing_parts(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, capability: Any
) -> None:
    """Injected host seconds test accounting, not NPU performance."""
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    clock = [0.0, 0.0]
    monkeypatch.setattr(sync_module, "perf_counter", lambda: clock[runtime.thread.rank])

    def instrument(engine: Any) -> Any:
        rank = engine.metadata.worker_id
        connector = engine.gpu_connector
        if capability is not None:
            monkeypatch.setattr(
                connector,
                "supports_layerwise_prefill_transfer_timings",
                capability,
                raising=False,
            )
        prepare = connector.prepare_layerwise_prefill_slots
        process = engine.token_database.process_tokens
        ack = engine.layerwise_prefill_ack
        transfer = connector.transfer_layerwise_prefill_row

        def prepare_slots(*args: Any, **kwargs: Any) -> Any:
            result = prepare(*args, **kwargs)
            clock[rank] += 0.002
            return result

        def plan_tokens(*args: Any, **kwargs: Any) -> Any:
            result = list(process(*args, **kwargs))
            clock[rank] += 0.001
            return result

        def acknowledge(*args: Any, **kwargs: Any) -> None:
            ack(*args, **kwargs)
            clock[rank] += 0.003

        def timed_transfer(
            *args: Any, kv_group: int, direction: bool, timing: dict
        ) -> None:
            assert len(args) == 5
            assert timing is backend._transfer_timings[int(direction)]
            transfer(*args, kv_group=kv_group, direction=direction)
            seconds = (0.008, 0.016, 0.032) if direction else (0.001, 0.002, 0.004)
            for key, value in zip(
                ("prepare_s", "submit_s", "fence_s"), seconds, strict=True
            ):
                timing[key] = timing.get(key, 0.0) + value
            clock[rank] += sum(seconds)

        monkeypatch.setattr(connector, "prepare_layerwise_prefill_slots", prepare_slots)
        monkeypatch.setattr(engine.token_database, "process_tokens", plan_tokens)
        monkeypatch.setattr(engine, "layerwise_prefill_ack", acknowledge)
        if capability is True:
            monkeypatch.setattr(
                connector, "transfer_layerwise_prefill_row", timed_transfer
            )
        # Non-literal/truthy capabilities retain a hook with no timing keyword.
        backend = engine.layerwise_prefill_window_backend
        assert backend._transfer_timing_enabled is (capability is True)
        return backend

    runtime.thread.rank = 0
    root_backend = instrument(root)
    runtime.thread.rank = 1
    passive_backend = instrument(passive)
    logged = []
    monkeypatch.setattr(
        sync_module.logger,
        "info",
        lambda message, *args: logged.append((runtime.thread.rank, message % args)),
    )
    steps = ((2, 0, 300), (1, 300, 530), (1, 530, 560))
    for step, (count, start, end) in enumerate(steps, start=1):
        requests = [_request(i, start=start, end=end) for i in range(count)]
        runtime.parallel(
            lambda requests=requests: _step(root, root_backend, requests),
            lambda requests=requests: _step(passive, passive_backend, requests),
        )
        assert len(logged) == 2 * step
        begin, line = [entry[1] for entry in logged[-2:]]
        assert f"event=begin step={step} requests={count}" in begin
        assert f"event=end step={step} requests={count} saved=(79, 22)" in line
        assert all("parts_ms" not in entry[1] for entry in logged[::2])
        assert (
            f"prepared_slots={4 * count} transfer_timing={capability is True}" in line
        )
        assert f"reused_rows={101 if start else 0} " in line
        assert f"published_handles={101 * count * (1 if start == 530 else 2)} " in line
        assert f"completed_ends={[(req.request_id, end) for req in requests]}" in line
        bind_parts = (4.0, count * 8.0, 3.0)
        assert f"bind_parts_ms=(plan,slots,ack):{bind_parts}" in line
        load_count = 101 if start and capability is True else 0
        save_count = 101 * count if capability is True else 0
        ack_ms = 101 * (1 + count) * 3.0
        load_parts = (load_count * 1.0, load_count * 2.0, load_count * 4.0, ack_ms)
        save_parts = (save_count * 8.0, save_count * 16.0, save_count * 32.0, ack_ms)
        assert f"load_parts_ms=(prepare,submit,fence,ack):{load_parts}" in line
        assert f"save_parts_ms=(prepare,submit,fence,ack):{save_parts}" in line
        for name, expected in (
            ("bind_ms", sum(bind_parts)),
            ("load_ms", sum(load_parts)),
            ("save_ms", sum(save_parts)),
            ("other_ms", 3.0),
            ("elapsed_ms", sum(bind_parts) + sum(load_parts) + sum(save_parts) + 3.0),
        ):
            actual = float(line.split(f" {name}=", 1)[1].split()[0])
            assert actual == pytest.approx(expected, abs=0.001)
    assert all(rank == 0 for rank, _ in logged)


def test_failed_step_does_not_log_a_completed_extent(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    logged = []
    monkeypatch.setattr(
        sync_module.logger, "info", lambda message, *args: logged.append(message % args)
    )
    backend.bind_step([_request()], _registry(engine))
    with pytest.raises(ValueError, match="Incomplete"):
        backend.finish_step()
    assert len(logged) == 1
    assert "event=begin" in logged[0]


@pytest.mark.parametrize("failure", ["missing", "passive_load", "root_store"])
def test_tp_errors_reach_all_peers(runtime: Any, failure: str) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend
    if failure == "passive_load":
        runtime.parallel(
            lambda: _step(root, rb, [_request()]),
            lambda: _step(passive, pb, [_request()]),
        )
        passive.gpu_connector.fail_load = True
    requests = (
        [_request(start=300, end=530)] if failure != "root_store" else [_request()]
    )
    runtime.parallel(
        lambda: rb.bind_step(requests, _registry(root)),
        lambda: pb.bind_step(requests, _registry(passive)),
    )
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    if failure == "root_store":
        runtime.parallel(
            lambda: rb.wait_for_load(metadata), lambda: pb.wait_for_load(metadata)
        )
        root.gpu_connector.fail_store = True

    def run(engine: Any, backend: Any) -> None:
        with pytest.raises(ValueError, match="Missing required|sentinel failure"):
            if failure == "root_store":
                backend.sync_save(metadata, _registry(engine)[metadata.row.layer_name])
            else:
                backend.wait_for_load(metadata)

    runtime.parallel(lambda: run(root, rb), lambda: run(passive, pb))
    rb.abort_request("req-0")
    pb.abort_request("req-0")


@pytest.mark.parametrize(
    "setting,value",
    [
        ("store_async", True),
        ("save_unfull_chunk", False),
        ("shared_cpu_cache_strict", False),
        ("chunk_size", 128),
        ("enable_shared_cpu_cache", False),
        ("dsa_two_groups", False),
    ],
)
def test_factory_rejects_unsupported_storage(
    runtime: Any, setting: str, value: Any
) -> None:
    engine = runtime.engine()
    setattr(engine.config, setting, value)
    with pytest.raises(ValueError, match="requires"):
        _ = engine.layerwise_prefill_window_backend


@pytest.mark.parametrize(
    "setting",
    [
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "graph",
        "page_first",
        "dtype",
        "hook",
    ],
)
def test_factory_rejects_unsupported_runtime(runtime: Any, setting: str) -> None:
    engine = runtime.engine()
    if setting == "graph":
        runtime.serving.model_config.enforce_eager = False
    elif setting == "page_first":
        engine.config.extra_config["mooncake_page_first_multi_buffer"] = True
    elif setting == "dtype":
        engine.metadata.kv_dtype = torch.float16
    elif setting == "hook":
        engine.gpu_connector.transfer_layerwise_prefill_row = None
    else:
        setattr(runtime.serving.parallel_config, setting, 2)
    with pytest.raises(ValueError):
        _ = engine.layerwise_prefill_window_backend


def test_factory_off_and_incomplete_step(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = runtime.engine()
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE")
    assert engine.layerwise_prefill_window_backend is None
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    backend = engine.layerwise_prefill_window_backend
    backend.bind_step([_request()], _registry(engine))
    with pytest.raises(ValueError, match="79/22"):
        backend.finish_step()
    backend.abort_request("req-0")


@pytest.mark.parametrize(
    "generation,failure",
    [(1, None), (2, None), (2, "missing"), (2, "revision"), (2, "slab")],
)
def test_real_two_process_gloo_shared_slab(
    runtime: Any,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    generation: int,
    failure: str | None,
) -> None:
    """Use real Gloo acknowledgements and shared slab bytes, not fake collectives."""
    root = runtime.engine(size=2)
    root.storage_manager.local_cpu_backend.memory_allocator.buffer.share_memory_()
    passive = runtime.engine(1, root, size=2)
    context = multiprocessing.get_context("fork")
    results = context.Queue()

    def worker(rank: int, engine: Any) -> None:
        try:
            torch.distributed.all_gather_object = runtime.real_gather
            torch.distributed.is_initialized = runtime.real_initialized
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{tmp_path}/gloo",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=30),
            )
            vllm.distributed.parallel_state.get_tp_group = lambda: SimpleNamespace(
                world_size=2,
                rank_in_group=rank,
                cpu_group=torch.distributed.group.WORLD,
            )

            def broadcast(obj: Any, src: int) -> Any:
                payload = [obj]
                torch.distributed.broadcast_object_list(payload, src=src)
                return payload[0]

            engine.broadcast_object_fn = broadcast
            backend = engine.layerwise_prefill_window_backend
            counts = _count_manifest_sources(monkeypatch, engine, backend)
            _step(engine, backend, [_request()])
            assert counts == Counter(
                plans=202,
                resolves=202,
                broadcasts=202,
                acks=406,
                **({"handles": 202} if rank == 0 else {"views": 202}),
            )
            old = dict(backend._prefixes)
            if failure is not None:
                requests = [_request(start=300, end=530, generation=generation)]
                backend.bind_step(requests, _registry(engine))
                identity = ("req-0", generation, 0, 0)
                prior = backend._prefixes[identity]
                if rank == 1:
                    if failure == "missing":
                        del backend._prefixes[identity]
                    elif failure == "revision":
                        backend._prefixes[identity] = replace(prior, revision=2)
                    else:
                        engine.shared_cpu_cache_generation = 12
                counts.clear()
                calls = list(engine.gpu_connector.calls)
                with pytest.raises(ValueError, match="manifest|identity mismatch"):
                    backend.wait_for_load(
                        _metadata(_view(), _view().rows_by_group[0][0], requests)
                    )
                assert counts == Counter(acks=1)
                assert engine.gpu_connector.calls == calls
                assert backend._failed
                backend._prefixes[identity] = prior
                backend.abort_request("req-0")
                results.put((rank, "closed"))
                torch.distributed.destroy_process_group()
                return
            counts.clear()
            load = backend.wait_for_load

            def warm_load(metadata: Any) -> None:
                row = metadata.row
                prior = old["req-0", 1, row.kv_group, row.row_ordinal]
                identity = ("req-0", generation, row.kv_group, row.row_ordinal)
                assert backend._prefixes[identity] is prior
                assert prior.revision == 1 and prior.slab_generation == 11
                ownership = [
                    (obj.get_ref_count(), obj.metadata.pin_count)
                    for obj in prior.objects
                ]
                before = counts.copy()
                load(metadata)
                assert counts - before == Counter(acks=2)
                assert backend._prefixes[identity] is prior
                assert ownership == [
                    (obj.get_ref_count(), obj.metadata.pin_count)
                    for obj in prior.objects
                ]
                assert engine.gpu_connector.calls[-1] == (
                    False,
                    row.kv_group,
                    (0, 256),
                    (256, 300),
                )

            monkeypatch.setattr(backend, "wait_for_load", warm_load)
            _step(
                engine, backend, [_request(start=300, end=530, generation=generation)]
            )
            assert counts == Counter(
                plans=101,
                resolves=101,
                broadcasts=101,
                acks=406,
                **({"handles": 202} if rank == 0 else {"views": 202}),
            )
            for (_, _, group, row), prior in old.items():
                current = backend._prefixes["req-0", generation, group, row]
                assert current.revision == 2 and current.slab_generation == 11
                assert len(current.objects) == 3
                assert current.objects[0] is prior.objects[0]
                assert prior.objects[0].get_ref_count() == (2 if rank == 0 else 1)
                assert prior.objects[0].metadata.pin_count == 1
                assert prior.objects[1].get_ref_count() == (1 if rank == 0 else 0)
                assert prior.objects[1].metadata.pin_count == 0
            backend.abort_request("req-0")
            calls = engine.gpu_connector.calls
            results.put(
                (
                    rank,
                    sum(call[0] for call in calls),
                    sum(not call[0] for call in calls),
                )
            )
            torch.distributed.destroy_process_group()
        except BaseException as exc:
            results.put((rank, str(exc)))
            raise

    processes = [
        context.Process(target=worker, args=(rank, engine))
        for rank, engine in enumerate((root, passive))
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert sorted(outcomes) == (
            [(0, 202, 101), (1, 0, 101)]
            if failure is None
            else [(0, "closed"), (1, "closed")]
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


def test_large_manifest_reuses_208_chunks_and_publishes_only_16(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TensorMemoryAllocator rounds each tiny sentinel chunk to a 4 KiB slab slot.
    root = runtime.engine(size=2, blocks=900, slab_bytes=128 << 20)
    passive = runtime.engine(1, root, size=2, blocks=900)
    runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend
    first_end, next_end = 208 * 256, 224 * 256
    req = replace(
        _request(),
        token_ids=tuple(range(next_end)),
        compute_end=first_end,
        block_ids_by_bank=tuple(
            (tuple(range(1 + bank * 448, 449 + bank * 448)),) * 2 for bank in (0, 1)
        ),
    )
    runtime.parallel(lambda: _step(root, rb, [req]), lambda: _step(passive, pb, [req]))
    previous = [backend._prefixes["req-0", 1, 0, 0] for backend in (rb, pb)]
    req = replace(
        req,
        compute_start=first_end,
        restore_end=first_end,
        compute_end=next_end,
        allocation_generation=2,
    )
    runtime.parallel(
        lambda: rb.bind_step([req], _registry(root)),
        lambda: pb.bind_step([req], _registry(passive)),
    )
    counts = [
        _count_manifest_sources(monkeypatch, engine, backend)
        for engine, backend in ((root, rb), (passive, pb))
    ]
    metadata = _metadata(_view(), _view().rows_by_group[0][0], [req])
    runtime.parallel(
        lambda: rb.wait_for_load(metadata), lambda: pb.wait_for_load(metadata)
    )
    assert counts == [Counter(acks=2), Counter(acks=2)]
    for engine in (root, passive):
        assert engine.gpu_connector.calls[-1] == (
            False,
            0,
            tuple(range(0, first_end, 256)),
            tuple(range(256, first_end + 1, 256)),
        )
        slots = _slots(req, 0, 0, next_end)
        for index, plane in enumerate(engine.gpu_connector.planes[0]):
            expected = _sentinel(req, 0, 0, index, next_end)
            assert torch.equal(plane.view(-1)[slots[:first_end]], expected[:first_end])
            plane.view(-1)[slots[first_end:]] = expected[first_end:]
    runtime.parallel(
        lambda: rb.sync_save(metadata, _registry(root)[metadata.row.layer_name]),
        lambda: pb.sync_save(metadata, _registry(passive)[metadata.row.layer_name]),
    )
    assert counts == [
        Counter(acks=4, plans=1, resolves=1, broadcasts=1, handles=16),
        Counter(acks=4, plans=1, resolves=1, broadcasts=1, views=16),
    ]
    for rank, (backend, prior) in enumerate(zip((rb, pb), previous, strict=True)):
        current = backend._prefixes["req-0", 2, 0, 0]
        assert len(current.objects) == 224
        assert all(
            obj is old
            for obj, old in zip(current.objects[:208], prior.objects, strict=True)
        )
        assert all(
            obj.get_ref_count() == (2 if rank == 0 else 1)
            and obj.metadata.pin_count == 1
            for obj in prior.objects
        )
        backend.abort_request("req-0")


@pytest.mark.parametrize(
    "phase,mutate",
    [
        ("bind", "missing"),
        ("load", "missing"),
        ("load", "revision"),
        ("load", "extent"),
        ("load", "count"),
        ("load", "slab"),
        ("load", "engine_slab"),
        ("load", "ready"),
        ("save", "missing"),
        ("save", "revision"),
        ("save", "slab"),
    ],
)
def test_one_peer_manifest_mismatch_fails_before_transfer_or_broadcast(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, phase: str, mutate: str
) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend
    runtime.parallel(
        lambda: _step(root, rb, [_request()]),
        lambda: _step(passive, pb, [_request()]),
    )
    requests = [_request(start=300, end=530, generation=2)]
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    if phase != "bind":
        runtime.parallel(
            lambda: rb.bind_step(requests, _registry(root)),
            lambda: pb.bind_step(requests, _registry(passive)),
        )
    if phase == "save":
        runtime.parallel(
            lambda: rb.wait_for_load(metadata), lambda: pb.wait_for_load(metadata)
        )
    identity = ("req-0", 1 if phase == "bind" else 2, 0, 0)
    prior = pb._prefixes[identity]
    ownership = [(obj.get_ref_count(), obj.metadata.pin_count) for obj in prior.objects]
    if mutate == "missing":
        del pb._prefixes[identity]
    elif mutate == "revision":
        pb._prefixes[identity] = replace(prior, revision=pb._step)
    elif mutate == "extent":
        pb._prefixes[identity] = replace(prior, ends=[256, 299])
    elif mutate == "count":
        pb._prefixes[identity] = replace(prior, objects=prior.objects[:1])
    elif mutate == "slab":
        pb._prefixes[identity] = replace(prior, slab_generation=12)
    elif mutate == "ready":
        pb._ready.add((0, 0))
    else:
        passive.shared_cpu_cache_generation = 12
    counts = [
        _count_manifest_sources(monkeypatch, engine, backend)
        for engine, backend in ((root, rb), (passive, pb))
    ]
    transfers = [list(engine.gpu_connector.calls) for engine in (root, passive)]

    def fail(engine: Any, backend: Any) -> None:
        with pytest.raises(ValueError, match="manifest|identity mismatch"):
            if phase == "bind":
                backend.bind_step(requests, _registry(engine))
            elif phase == "load":
                backend.wait_for_load(metadata)
            else:
                backend.sync_save(metadata, _registry(engine)[metadata.row.layer_name])

    try:
        runtime.parallel(lambda: fail(root, rb), lambda: fail(passive, pb))
        assert counts == [Counter(acks=1), Counter(acks=1)]
        assert transfers == [engine.gpu_connector.calls for engine in (root, passive)]
        assert ownership == [
            (obj.get_ref_count(), obj.metadata.pin_count) for obj in prior.objects
        ]
    finally:
        pb._prefixes[identity] = prior
        rb.abort_request("req-0")
        pb.abort_request("req-0")


@pytest.mark.parametrize("failure", ["future", "passive_ack"])
def test_delta_failure_preserves_old_manifest_and_releases_only_fresh_ownership(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    root.config.remote_url = passive.config.remote_url = "mooncakestore://required"
    completed = Future()
    completed.set_result(None)
    remote = RequiredStore(root.storage_manager.local_cpu_backend, completed)
    root.storage_manager.storage_backends["RemoteBackend"] = remote
    runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend
    runtime.parallel(
        lambda: _step(root, rb, [_request()]),
        lambda: _step(passive, pb, [_request()]),
    )
    requests = [_request(start=300, end=530, generation=2)]
    runtime.parallel(
        lambda: rb.bind_step(requests, _registry(root)),
        lambda: pb.bind_step(requests, _registry(passive)),
    )
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    runtime.parallel(
        lambda: rb.wait_for_load(metadata), lambda: pb.wait_for_load(metadata)
    )
    identity = ("req-0", 2, 0, 0)
    previous = [backend._prefixes[identity] for backend in (rb, pb)]
    ownership = [
        [(obj.get_ref_count(), obj.metadata.pin_count) for obj in prior.objects]
        for prior in previous
    ]
    cached = dict(root.storage_manager.local_cpu_backend.hot_cache)
    fresh_views = []
    create_view = passive.shared_cpu_cache_passive_allocator.create_view

    def acquire(*args: Any, **kwargs: Any) -> Any:
        obj = create_view(*args, **kwargs)
        fresh_views.append(obj)
        return obj

    monkeypatch.setattr(
        passive.shared_cpu_cache_passive_allocator, "create_view", acquire
    )
    if failure == "passive_ack":
        ack = passive.layerwise_prefill_ack

        def reject(identity: Any, error: Any = None, *, flush: bool = True) -> None:
            if identity[1][0] == "save":
                error = ValueError("save acknowledgement failure")
            ack(identity, error, flush=flush)

        monkeypatch.setattr(passive, "layerwise_prefill_ack", reject)
    delayed = Future()
    remote.put_result = [delayed]
    remote.submitted.clear()

    def save(engine: Any, backend: Any) -> None:
        with pytest.raises(
            ValueError, match="delta persist failed|acknowledgement failure"
        ):
            backend.sync_save(metadata, _registry(engine)[metadata.row.layer_name])

    with ThreadPoolExecutor(max_workers=1) as pool:
        saved = pool.submit(
            runtime.parallel, lambda: save(root, rb), lambda: save(passive, pb)
        )
        assert remote.submitted.wait(5)
        assert not saved.done()
        assert len(remote.calls[-1][0]) == 2
        if failure == "future":
            delayed.set_exception(RuntimeError("delta persist failed"))
        else:
            delayed.set_result(None)
        saved.result(timeout=20)
    for backend, prior, owned in zip((rb, pb), previous, ownership, strict=True):
        assert backend._failed
        assert backend._prefixes[identity] is prior
        assert prior.revision == 1 and prior.slab_generation == 11
        assert owned == [
            (obj.get_ref_count(), obj.metadata.pin_count) for obj in prior.objects
        ]
    fresh_root = [
        obj
        for key, obj in root.storage_manager.local_cpu_backend.hot_cache.items()
        if key not in cached
    ]
    assert len(fresh_root) == 2
    assert all(
        obj.get_ref_count() == 1 and obj.metadata.pin_count == 0 for obj in fresh_root
    )
    assert len(fresh_views) == (0 if failure == "future" else 2)
    assert all(
        obj.get_ref_count() == 0 and obj.metadata.pin_count == 0 for obj in fresh_views
    )
    rb.abort_request("req-0")
    pb.abort_request("req-0")


def test_allocation_failure_releases_partial_batch(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    registry = _registry(engine)
    requests = [_request()]
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    backend.wait_for_load(metadata)
    allocate = engine.storage_manager.allocate
    acquired = []

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        if acquired:
            return None
        acquired.append(allocate(*args, **kwargs))
        return acquired[0]

    monkeypatch.setattr(engine.storage_manager, "allocate", fail_second)
    with pytest.raises(ValueError, match="allocation failed"):
        backend.sync_save(metadata, registry[metadata.row.layer_name])
    assert not acquired[0].is_valid()
    assert not engine.gpu_connector.calls
    backend.abort_request("req-0")


@pytest.mark.parametrize("change", ["tokens", "config", "generation", "frontier"])
def test_bind_rejects_changed_continuation(runtime: Any, change: str) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request(generation=2)])
    req = _request(start=300, end=530, generation=2)
    if change == "tokens":
        req = replace(req, token_ids=(99, *req.token_ids[1:]))
    elif change == "config":
        req = replace(req, request_configs={"lmcache.tag.tenant": "other"})
    elif change == "generation":
        req = replace(req, allocation_generation=1)
    else:
        req = replace(req, compute_start=299, restore_end=299)
    with pytest.raises(ValueError, match="retained prefix"):
        backend.bind_step([req], _registry(engine))
    backend.abort_request("req-0")


@pytest.mark.parametrize("generation", [1, 2])
def test_rebind_prepares_new_plans_and_restores_into_current_blocks(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, generation: int
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    snapshots = []
    finish = backend.finish_step

    def retain_plans() -> None:
        snapshots.append(dict(backend._slots))
        finish()
        assert not backend._slots

    monkeypatch.setattr(backend, "finish_step", retain_plans)
    _step(engine, backend, [_request()])
    req = _request(start=300, end=530, generation=generation)
    req = replace(
        req,
        block_ids_by_bank=tuple(
            tuple(
                tuple(block + (24 if generation == 2 else 0) for block in blocks)
                for blocks in groups
            )
            for groups in req.block_ids_by_bank
        ),
    )
    _step(engine, backend, [req])
    assert len(engine.gpu_connector.prepared) == 8
    for identity, old in snapshots[0].items():
        current = snapshots[1][identity]
        assert current is not old
        assert old.snapshot.numel() == 300 and current.snapshot.numel() == 530
        assert torch.equal(
            current.snapshot, _slots(req, identity[1], identity[2], req.compute_end)
        )
    assert all(engine.gpu_connector.plan_uses[i, False] == 0 for i in range(4))
    assert sum(engine.gpu_connector.plan_uses[i, False] for i in range(4, 8)) == 101
    backend.abort_request("req-0")
    with pytest.raises(ValueError, match="Released"):
        backend.bind_step([req], _registry(engine))


def test_constructor_requires_explicit_serving_config(runtime: Any) -> None:
    engine = runtime.engine()
    engine.configure_layerwise_prefill_sync(None)
    with pytest.raises(ValueError, match="configure_layerwise_prefill_sync"):
        LayerwisePrefillSyncBackend(engine, _view())


def test_tp_fence_failure_is_fatal_on_every_rank(runtime: Any) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)

    def acknowledge(engine: Any, error: Exception | None) -> None:
        with pytest.raises(LayerwisePrefillFenceError, match="device fence failed"):
            engine.layerwise_prefill_ack("row", error)

    runtime.parallel(
        lambda: acknowledge(root, None),
        lambda: acknowledge(passive, LayerwisePrefillFenceError("device fence failed")),
    )


def test_window_sync_save_forwards_request_metadata() -> None:
    seen = []

    def save(*args: Any, metadata: Any) -> None:
        seen.append(metadata)

    ops = SimpleNamespace(layout_signature=lambda: "sync", sync_save=save)
    backend = LayerwisePrefillNPUWindowBackend(_view(), ops)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], [_request()])
    backend.sync_save(metadata, [torch.zeros(1), torch.zeros(1)])
    assert seen == [metadata]


@pytest.mark.parametrize(
    "failure",
    [
        "omitted",
        "disconnected",
        "bypassed",
        "frozen",
        "no_completion",
        "none_futures",
        "empty_futures",
        "malformed_futures",
        "other_allocator",
        "single_key_connector",
    ],
)
def test_required_remote_never_silently_persists(runtime: Any, failure: str) -> None:
    engine = runtime.engine()
    engine.config.remote_url = "mooncakestore://required"
    manager = engine.storage_manager
    completed = Future()
    completed.set_result(None)
    remote = RequiredStore(manager.local_cpu_backend, completed)
    if failure != "omitted":
        manager.storage_backends["RemoteBackend"] = remote
    if failure == "disconnected":
        remote.connection = None
    elif failure == "bypassed":
        manager.set_backend_bypass("RemoteBackend", True)
    elif failure == "frozen":
        manager.set_freeze(True)
    elif failure == "no_completion":
        remote.completion = False
    elif failure == "none_futures":
        remote.put_result = None
    elif failure == "empty_futures":
        remote.put_result = []
    elif failure == "malformed_futures":
        remote.put_result = [None, completed]
    elif failure == "other_allocator":
        remote.cpu = object()
    elif failure == "single_key_connector":
        remote.connection.support_batched_put = lambda: False
    backend = engine.layerwise_prefill_window_backend
    requests = [_request()]
    registry = _registry(engine)
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    backend.wait_for_load(metadata)
    with pytest.raises(ValueError, match="RemoteBackend"):
        backend.sync_save(metadata, registry[metadata.row.layer_name])
    with pytest.raises(ValueError, match="Incomplete"):
        backend.finish_step()
    backend.abort_request("req-0")
    # Only LocalCPU's cache references may remain, never lent/manifest refs.
    assert all(
        obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
        for obj in manager.local_cpu_backend.hot_cache.values()
    )


@pytest.mark.parametrize(
    "failure", ["failed_future", "false_result", "malformed_future", "later_submit"]
)
def test_strict_put_drains_sibling_futures_before_consuming_refs(
    runtime: Any, failure: str
) -> None:
    engine = runtime.engine()
    manager = engine.storage_manager
    first, delayed = Future(), Future()
    if failure == "false_result":
        first.set_result(False)
    else:
        first.set_exception(RuntimeError("put failed"))
    remote = RequiredStore(manager.local_cpu_backend, first)
    remote.put_result = [first, delayed]
    if failure == "malformed_future":
        remote.put_result.insert(1, None)
    manager.storage_backends["RemoteBackend"] = remote
    if failure == "later_submit":
        first = Future()
        first.set_result(None)
        remote.put_result = [first, delayed]
        later = RequiredStore(manager.local_cpu_backend, first)

        def fail_submit(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("later submit failed")

        later.batched_submit_put_task = fail_submit
        manager.storage_backends["LaterBackend"] = later
    obj = manager.allocate(torch.Size([16]), torch.bfloat16)
    key = next(engine.token_database.process_tokens(list(range(16))))[2].get_layer(0)
    # Hold a separate test reference to verify the API consumes exactly its loan.
    obj.ref_count_up()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            manager.batched_put_sync_required,
            [key],
            [obj],
            required_backends=("RemoteBackend",),
        )
        assert remote.submitted.wait(5)
        assert not result.done()
        assert obj.get_ref_count() == 3  # test + loan + LocalCPU
        delayed.set_result(None)
        with pytest.raises(RuntimeError):
            result.result(timeout=5)
    assert obj.get_ref_count() == 2
    obj.ref_count_down()


def test_strict_put_consumes_validation_error_loan(runtime: Any) -> None:
    manager = runtime.engine().storage_manager
    obj = manager.allocate(torch.Size([16]), torch.bfloat16)
    obj.ref_count_up()
    with pytest.raises(ValueError, match="matching keys"):
        manager.batched_put_sync_required(
            [], [obj], required_backends=("RemoteBackend",)
        )
    assert obj.get_ref_count() == 1
    obj.ref_count_down()


@pytest.mark.parametrize("retry_fails", [False, True])
def test_failed_remote_local_bootstrap_republishes_full_prefix(
    runtime: Any, retry_fails: bool
) -> None:
    engine = runtime.engine()
    engine.config.remote_url = "mooncakestore://required"
    failed = Future()
    failed.set_exception(RuntimeError("first remote put failed"))
    remote = RequiredStore(engine.storage_manager.local_cpu_backend, failed)
    engine.storage_manager.storage_backends["RemoteBackend"] = remote
    backend = engine.layerwise_prefill_window_backend
    requests, registry = [_request()], _registry(engine)
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    backend.wait_for_load(metadata)
    with pytest.raises(ValueError, match="first remote put failed"):
        backend.sync_save(metadata, registry[metadata.row.layer_name])
    backend.abort_request("req-0")
    cached = dict(engine.storage_manager.local_cpu_backend.hot_cache)
    assert len(cached) == 2  # Full [0,256) plus partial [256,300).
    original_ids = tuple(id(obj) for obj in cached.values())
    assert all(obj.metadata.pin_count == 0 for obj in cached.values())

    delayed = Future()
    remote.put_result = [delayed]
    remote.submitted.clear()
    calls = list(engine.gpu_connector.calls)
    requests = [_request(start=300, end=530, generation=2)]
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    with ThreadPoolExecutor(max_workers=1) as pool:
        loaded = pool.submit(backend.wait_for_load, metadata)
        assert remote.submitted.wait(5)
        assert not loaded.done()
        assert remote.calls[-1] == (tuple(cached), original_ids)
        assert engine.gpu_connector.calls == calls  # No H2D before commit, no D2H.
        if retry_fails:
            delayed.set_exception(RuntimeError("bootstrap commit failed"))
            with pytest.raises(ValueError, match="bootstrap commit failed"):
                loaded.result(timeout=5)
        else:
            delayed.set_result(None)
            loaded.result(timeout=5)
    if retry_fails:
        assert engine.gpu_connector.calls == calls
        assert all(
            obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
            for obj in cached.values()
        )
        backend.abort_request("req-0")
        return
    assert engine.gpu_connector.calls[-1] == (False, 0, (0, 256), (256, 300))
    assert all(obj.metadata.pin_count == 1 for obj in cached.values())
    backend.abort_request("req-0")
    assert all(obj.metadata.pin_count == 0 for obj in cached.values())


def test_committed_own_prefix_does_not_republish_unchanged_chunks(runtime: Any) -> None:
    engine = runtime.engine()
    engine.config.remote_url = "mooncakestore://required"
    committed = Future()
    committed.set_result(None)
    remote = RequiredStore(engine.storage_manager.local_cpu_backend, committed)
    engine.storage_manager.storage_backends["RemoteBackend"] = remote
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request()])
    assert len(remote.calls) == 101
    remote.calls.clear()
    requests = [_request(start=300, end=530)]
    registry = _registry(engine)
    backend.bind_step(requests, registry)
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    backend.wait_for_load(metadata)
    assert remote.calls == []
    backend.sync_save(metadata, registry[metadata.row.layer_name])
    assert len(remote.calls) == 1 and len(remote.calls[0][0]) == 2
    backend.abort_request("req-0")


@pytest.mark.parametrize("failure", ["disconnected", "empty_futures", "failed_future"])
def test_required_remote_failure_reaches_passive(runtime: Any, failure: str) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    root.config.remote_url = passive.config.remote_url = "mooncakestore://required"
    future = Future()
    future.set_exception(RuntimeError("required put failed"))
    remote = RequiredStore(root.storage_manager.local_cpu_backend, future)
    root.storage_manager.storage_backends["RemoteBackend"] = remote
    if failure == "disconnected":
        remote.connection = None
    elif failure == "empty_futures":
        remote.put_result = []
    runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend
    requests = [_request()]
    runtime.parallel(
        lambda: rb.bind_step(requests, _registry(root)),
        lambda: pb.bind_step(requests, _registry(passive)),
    )
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    runtime.parallel(
        lambda: rb.wait_for_load(metadata), lambda: pb.wait_for_load(metadata)
    )

    def save(engine: Any, backend: Any) -> None:
        with pytest.raises(ValueError, match="RemoteBackend|required put failed"):
            backend.sync_save(metadata, _registry(engine)[metadata.row.layer_name])

    runtime.parallel(lambda: save(root, rb), lambda: save(passive, pb))
    assert not passive.gpu_connector.calls
    rb.abort_request("req-0")
    pb.abort_request("req-0")


def test_live_prefix_pin_leases_survive_timeout_and_cleanup_once(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    monitor = PinMonitor.GetOrCreate()
    _step(engine, backend, [_request()])
    cpu = engine.storage_manager.local_cpu_backend
    old = list(cpu.hot_cache.values())
    now = [pin_monitor_module.time.time() + 1_000_000]
    monkeypatch.setattr(pin_monitor_module.time, "time", lambda: now[0])
    assert monitor._check_timeouts()[2] == 0
    assert all(obj.metadata.pin_count == 1 for obj in old)
    _step(engine, backend, [_request(start=300, end=530)])
    now[0] += 1_000_000
    assert monitor._check_timeouts()[2] == 0
    assert all(obj.metadata.pin_count <= 1 for obj in cpu.hot_cache.values())
    assert sum(obj.metadata.pin_count for obj in cpu.hot_cache.values()) == 303
    backend.abort_request("req-0")
    backend.abort_request("req-0")
    assert all(obj.metadata.pin_count == 0 for obj in cpu.hot_cache.values())
    ordinary = next(iter(cpu.hot_cache.values()))
    ordinary.pin()
    now[0] += 1_000_000
    assert monitor._check_timeouts()[2] == 1
    assert ordinary.metadata.pin_count == 0
    with pytest.raises(ValueError, match="no protected pin"):
        monitor.release_pin_lease(ordinary)
    assert ordinary.metadata.pin_count == 0


def test_pin_leases_are_balanced_for_shared_objects(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = runtime.engine().storage_manager
    obj = manager.allocate(torch.Size([16]), torch.bfloat16)
    monitor = PinMonitor.GetOrCreate()
    with monitor.protect_pins() as leases:
        obj.pin()
        obj.pin()
        leases.extend([obj, obj])
    now = pin_monitor_module.time.time() + 1_000_000
    monkeypatch.setattr(pin_monitor_module.time, "time", lambda: now)
    assert monitor._check_timeouts()[2] == 0
    monitor.release_pin_lease(obj)
    assert obj.metadata.pin_count == 1
    assert monitor._check_timeouts()[2] == 0
    monitor.release_pin_lease(obj)
    assert obj.metadata.pin_count == 0
    with pytest.raises(ValueError):
        monitor.release_pin_lease(obj)
    obj.ref_count_down()


def test_bootstrap_io_does_not_block_timeout_reclamation(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    requests = [_request(start=300, end=530)]
    backend.bind_step(requests, _registry(engine))
    metadata = _metadata(_view(), _view().rows_by_group[0][0], requests)
    monitor = PinMonitor.GetOrCreate()
    manager = engine.storage_manager
    ordinary = manager.allocate(torch.Size([16]), torch.bfloat16)
    ordinary.pin()
    now = pin_monitor_module.time.time() + 1_000_000
    monkeypatch.setattr(pin_monitor_module.time, "time", lambda: now)
    fetching, proceed = Event(), Event()
    get = manager.batched_get

    def delayed_get(*args: Any, **kwargs: Any) -> Any:
        fetching.set()
        assert proceed.wait(5)
        return get(*args, **kwargs)

    monkeypatch.setattr(manager, "batched_get", delayed_get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        loaded = pool.submit(backend.wait_for_load, metadata)
        assert fetching.wait(5)
        try:
            swept = pool.submit(monitor._check_timeouts)
            assert swept.result(timeout=2)[2] == 1
            assert ordinary.metadata.pin_count == 0
        finally:
            proceed.set()
        loaded.result(timeout=5)
    ordinary.ref_count_down()
    backend.abort_request("req-0")


def test_selected_timeout_cannot_race_lease_adoption_or_release(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = runtime.engine().storage_manager
    monitor = PinMonitor.GetOrCreate()
    obj = manager.allocate(torch.Size([16]), torch.bfloat16)
    obj.pin()
    now = [pin_monitor_module.time.time() + 1_000_000]
    monkeypatch.setattr(pin_monitor_module.time, "time", lambda: now[0])
    selected, proceed = Event(), Event()
    force = monitor._force_unpin_timeout_object

    def delayed_force(*args: Any) -> bool:
        selected.set()
        assert proceed.wait(5)
        return force(*args)

    monkeypatch.setattr(monitor, "_force_unpin_timeout_object", delayed_force)
    with ThreadPoolExecutor(max_workers=1) as pool:
        swept = pool.submit(monitor._check_timeouts)
        assert selected.wait(5)
        with monitor.protect_pins() as leases:
            obj.pin()
            leases.append(obj)
        proceed.set()
        assert swept.result(timeout=5)[2] == 0
    assert obj.metadata.pin_count == 2
    now[0] += 1_000_000
    monitor.release_pin_lease(obj)
    # Even a stale candidate obtained before release cannot expire the ordinary
    # remaining pin until its newly restored timeout elapses.
    assert force(obj, 1_000_000) is False
    assert obj.metadata.pin_count == 1
    now[0] += 1_000_000
    assert force(obj, 1_000_000) is True
    assert obj.metadata.pin_count == 0
    obj.ref_count_down()


@pytest.mark.parametrize(
    "value,enabled",
    [("true", True), (" TRUE ", True), ("false", False), (" False ", False)],
)
def test_p_flag_strict_valid_spellings(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, value: str, enabled: bool
) -> None:
    engine = runtime.engine()
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", value)
    assert (engine.layerwise_prefill_window_backend is not None) is enabled


@pytest.mark.parametrize("value", ["1", "0", "yes", "no", "on", "off", "", "tru"])
def test_p_flag_rejects_non_boolean_spellings(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    engine = runtime.engine()
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", value)
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        _ = engine.layerwise_prefill_window_backend
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        LayerwisePrefillSyncBackend(engine, _view())


@pytest.mark.parametrize(
    "method,tokens,supported",
    [
        ("mtp", 1, True),
        ("mtp", 2, False),
        ("mtp", 0, False),
        ("eagle", 1, False),
        ("mtp", True, False),
    ],
)
def test_speculation_cannot_repeat_physical_row(
    runtime: Any, method: str, tokens: Any, supported: bool
) -> None:
    engine = runtime.engine()
    runtime.serving.speculative_config = SimpleNamespace(
        method=method, num_speculative_tokens=tokens
    )
    if supported:
        assert engine.layerwise_prefill_window_backend.supports_sync_callbacks
    else:
        with pytest.raises(ValueError, match="num_speculative_tokens=1"):
            _ = engine.layerwise_prefill_window_backend
