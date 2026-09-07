# SPDX-License-Identifier: Apache-2.0
"""CPU sentinels exercise production P manifests, storage and TP handle exchange."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from queue import Queue
from threading import Barrier, Event, Lock, local
from types import SimpleNamespace
from typing import Any
import multiprocessing

# Third Party
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
from lmcache.v1.memory_management import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import ChunkedTokenDatabase
import lmcache.v1.config as lmcache_config
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


class CPUConnector:
    """Exact packed-plane transfer, with all rows aliasing two recyclable banks."""

    def __init__(self) -> None:
        self.layouts = {}
        self.calls = []
        self.fail_load = False
        self.fail_store = False
        self.planes = {
            group: [
                torch.full((64, 128, 1, 1), -100, dtype=torch.bfloat16)
                for _ in range(2 if group == 0 else 1)
            ]
            for group in (0, 1)
        }

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        self.kvcaches = kwargs["kvcaches"]

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

    def transfer_layerwise_prefill_row(
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
    real_initialized = torch.distributed.is_initialized

    def tp() -> Any:
        return SimpleNamespace(
            world_size=world.size,
            rank_in_group=getattr(thread, "rank", 0),
            cpu_group=barrier,
        )

    def gather(out: list, status: Any, group: Any) -> None:
        assert group is barrier
        statuses[thread.rank] = deepcopy(status)
        barrier.wait()
        out[:] = statuses
        barrier.wait()

    monkeypatch.setattr(vllm.distributed.parallel_state, "get_tp_group", tp)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    def engine(rank: int = 0, root: Any = None, size: int = 1) -> Any:
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
        result.gpu_connector = CPUConnector()
        result.token_database = ChunkedTokenDatabase(config, metadata)

        def broadcast(obj: Any, src: int) -> Any:
            assert src == 0
            if rank == 0:
                channel.put(deepcopy(obj))
                return obj
            return channel.get(timeout=10)

        result.broadcast_object_fn = broadcast
        if rank == 0:
            allocator = TensorMemoryAllocator(torch.empty(8 << 20, dtype=torch.uint8))
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


def test_tp_passive_restores_every_row_but_never_stores(runtime: Any) -> None:
    root = runtime.engine(size=2)
    passive = runtime.engine(1, root, size=2)
    runtime.thread.rank = 0
    root_backend = root.layerwise_prefill_window_backend
    runtime.thread.rank = 1
    passive_backend = passive.layerwise_prefill_window_backend
    for start, end in ((0, 300), (300, 530)):
        requests = [_request(index, start, end) for index in range(4)]
        runtime.parallel(
            lambda requests=requests: _step(root, root_backend, requests),
            lambda requests=requests: _step(passive, passive_backend, requests),
        )
    assert len(passive.gpu_connector.calls) == 4 * 101
    assert all(not call[0] for call in passive.gpu_connector.calls)
    assert sum(call[0] for call in root.gpu_connector.calls) == 8 * 101
    for index in range(4):
        root_backend.abort_request(f"req-{index}")
        passive_backend.abort_request(f"req-{index}")
    assert all(
        obj.metadata.pin_count == 0
        for obj in root.storage_manager.local_cpu_backend.hot_cache.values()
    )


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


def test_real_two_process_gloo_shared_slab(runtime: Any, tmp_path: Any) -> None:
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
            _step(engine, backend, [_request()])
            _step(engine, backend, [_request(start=300, end=530, generation=2)])
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
        assert sorted(outcomes) == [(0, 202, 101), (1, 0, 101)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


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


def test_new_allocation_restores_into_rebound_blocks(runtime: Any) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request()])
    req = _request(start=300, end=530, generation=2)
    req = replace(
        req,
        block_ids_by_bank=tuple(
            tuple(tuple(block + 24 for block in blocks) for blocks in groups)
            for groups in req.block_ids_by_bank
        ),
    )
    _step(engine, backend, [req])
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
