# SPDX-License-Identifier: Apache-2.0
"""Managed async backend integration, separate from the generic window tests."""

# ruff: noqa: F811

# Standard
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import timedelta
from threading import RLock
from types import SimpleNamespace
from typing import Any
import ast
import multiprocessing
import os
import traceback

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LayerwisePrefillWindowCoordinator,
    LMCacheConnectorMetadata,
    LoadSpec,
    ReqMeta,
    SaveSpec,
)
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.memory_management import TensorMemoryAllocator
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.scheduler import Scheduler
import pytest
import torch
import vllm.distributed.parallel_state

# First Party
from lmcache_ascend.v1.layerwise_prefill_async import LayerwisePrefillAsyncBackend
import lmcache_ascend.v1.layerwise_prefill_sync as sync_module

# Local
from tests.v1.test_layerwise_prefill_async import (
    DeferredCPUConnector,
    _callbacks,
    _compute,
)
from tests.v1.test_layerwise_prefill_sync import (
    _registry,
    _request,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_adapter import _make_adapter
from tests.v1.test_layerwise_prefill_sync_pages import (
    _assert_remote,
    page_runtime,  # noqa: F401
)


@pytest.fixture
def async_adapter(page_runtime: Any, monkeypatch: Any) -> Any:
    return _make_async_adapter(page_runtime, monkeypatch)


def _make_async_adapter(page_runtime: Any, monkeypatch: Any) -> Any:
    adapter = _make_adapter(page_runtime, monkeypatch)
    # The bookkeeping helper stubs telemetry. Cold remote reads must see no
    # active retrieve span, rather than a Mock-valued detailed-metrics mapping.
    LMCStatsMonitor.GetOrCreate().get_current_retrieve_stats.return_value = None
    # Reuse worker bookkeeping, but let the real factory construct a fresh
    # backend. Do not replace the sync helper's already-frozen backend.
    engine = page_runtime.engine()
    engine.config.extra_config["layerwise_prefill_transfer_window"] = True
    engine.gpu_connector = DeferredCPUConnector()
    page_runtime.serving.scheduler_config = SchedulerConfig.default_factory(
        async_scheduling=False
    )
    assert page_runtime.serving.scheduler_config.scheduler_cls is None
    assert page_runtime.serving.scheduler_config.get_scheduler_cls() is Scheduler
    page_runtime.serving.parallel_config.distributed_executor_backend = "mp"
    engine.configure_layerwise_prefill_sync(None)
    engine.is_store_async = False
    engine._direct_store_states = {}
    engine._engine_state_lock = RLock()
    engine.lookup_pins = {}
    engine.async_loading = None
    engine._shared_cpu_request_leases = {}
    adapter._manager.lmcache_engine = engine
    adapter._manager.lmcache_engine_metadata = engine.metadata
    adapter.config = engine.config
    adapter._layerwise_prefill_window = adapter._build_layerwise_prefill_window()
    adapter.kv_caches = _registry(engine)
    assert type(adapter._layerwise_prefill_backend) is LayerwisePrefillAsyncBackend
    assert engine.layerwise_prefill_window_backend is adapter._layerwise_prefill_backend
    assert isinstance(
        adapter._layerwise_prefill_window, LayerwisePrefillWindowCoordinator
    )
    assert adapter._layerwise_prefill_window.manages_pending_work
    assert adapter.supports_layerwise_prefill_transfer_window
    return adapter


def _start(adapter: Any, requests: list, *, final: bool = False) -> tuple:
    adapter._parent._connector_metadata = LMCacheConnectorMetadata(
        requests=[
            ReqMeta(
                req_id=req.request_id,
                token_ids=list(req.token_ids[: req.compute_end]),
                load_spec=LoadSpec(0, req.restore_end, bool(req.restore_end)),
                save_spec=SaveSpec(req.compute_start, True, True, True),
                is_last_prefill=final,
                request_configs=req.request_configs,
            )
            for req in requests
        ],
        layerwise_prefill_requests=requests,
    )
    callbacks = _callbacks(list(reversed(requests)))
    attention = {}
    for metadata in callbacks:
        execution = metadata.row.execution_ordinal
        attention[metadata.execution.latent.layer_name] = SimpleNamespace(
            layerwise_prefill_callback_metadata=tuple(
                m for m in callbacks if m.row.execution_ordinal == execution
            )
        )
    adapter.start_load_kv(SimpleNamespace(attn_metadata=attention))
    return callbacks


@pytest.mark.parametrize("request_count", [1, 4])
def test_adapter_101_rows_do_not_wait_assembly_at_eight_jobs(
    async_adapter: Any, page_runtime: Any, monkeypatch: Any, request_count: int
) -> None:
    adapter = async_adapter
    engine, window = adapter.lmcache_engine, adapter._layerwise_prefill_window
    requests = [_request(i) for i in range(request_count)]
    result = Future.result
    for start, end in ((0, 300), (300, 530)):
        if start:
            requests = [
                replace(
                    req,
                    compute_start=start,
                    restore_end=start,
                    compute_end=end,
                    allocation_generation=2,
                )
                for req in requests
            ]
        callbacks = _start(adapter, requests, final=bool(start))
        before = len(page_runtime.store.puts)

        def ready_result(future: Future, *args: Any, **kwargs: Any) -> Any:
            assert future.done(), "Coordinator blocked finite page assembly on itself"
            return result(future, *args, **kwargs)

        with monkeypatch.context() as assembly:
            assembly.setattr(Future, "result", ready_result)
            for execution in range(79):
                current = [m for m in callbacks if m.row.execution_ordinal == execution]
                for metadata in current:
                    adapter.wait_for_layerwise_prefill_load(metadata)
                for metadata in current:
                    _compute(engine, requests, metadata)
                    adapter.submit_layerwise_prefill_save(
                        metadata, adapter.kv_caches[metadata.row.layer_name]
                    )
                adapter.submit_layerwise_prefill_load(current[0])
                for metadata in current:
                    adapter.finish_layerwise_prefill_save(metadata)
                assert (
                    window.pending_jobs()
                    == window.pending_bytes()
                    == window.pending_futures()
                    == 0
                )
                assert all(
                    not adapter.layerwise_prefill_request_persist_done(req.request_id)
                    for req in requests
                )
        assert len(page_runtime.store.puts) == before
        adapter.wait_for_save()
        assert all(
            adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
        for req in requests:
            _assert_remote(engine, req, page_runtime.store)
    assert adapter.get_completed_decode_window_saves() == {
        req.request_id: 512 for req in requests
    }
    adapter.get_finished({req.request_id for req in requests})


@pytest.mark.parametrize("phase", ["save", "load", "finish"])
def test_coordinator_validation_reaches_backend_post_hcom(
    async_adapter: Any, phase: str
) -> None:
    adapter = async_adapter
    requests = [_request()]
    callbacks = _start(adapter, requests)
    current = [m for m in callbacks if m.row.execution_ordinal == 0]
    for metadata in current:
        adapter.wait_for_layerwise_prefill_load(metadata)
    metadata = current[0]
    bad = replace(metadata, request_generations=((requests[0].request_id, 2),))
    for member in current:
        adapter.submit_layerwise_prefill_save(
            bad if phase == "save" and member is metadata else member,
            adapter.kv_caches[member.row.layer_name],
        )
    adapter.submit_layerwise_prefill_load(bad if phase == "load" else metadata)
    # The model reaches projection/HCOM before coordinator errors are surfaced.
    with pytest.raises(ValueError, match="generation|identity|binding"):
        adapter.finish_layerwise_prefill_save(bad if phase == "finish" else metadata)
    adapter._layerwise_prefill_backend.abort_step()


def _assert_window_end_log(line: str, stats: dict, remote_jobs: int) -> None:
    assert "[PREFILL_SYNC_STEP] event=end" in line and "saved=(79, 22)" in line
    assert ast.literal_eval(line.split(" window_stats=", 1)[1]) == stats
    for resource, limit in (("jobs", 8), ("bytes", 64 << 20), ("futures", 8)):
        assert stats[f"max_{resource}"] == limit
        assert stats[f"pending_{resource}"] == 0
        assert 0 < stats[f"peak_{resource}"] <= limit
    assert stats["device_jobs"] == 101
    assert stats["remote_jobs"] == remote_jobs == 2
    assert stats["actual_jobs"] == 101 + remote_jobs


def _gloo_async_worker(
    rank: int,
    slab: torch.Tensor,
    rendezvous: str,
    fail: bool,
    draining: Any,
    allow_drain: Any,
    source_done: Any,
    released: Any,
    results: Any,
) -> None:
    """Spawn target: all engines, transport loops and fake stores are child-local."""
    try:
        torch.set_num_threads(1)
        with ExitStack() as stack:
            monkeypatch = stack.enter_context(pytest.MonkeyPatch.context())
            local = stack.enter_context(
                contextmanager(runtime.__wrapped__)(monkeypatch)
            )
            make_engine = local.engine
            root_mapping = SimpleNamespace(
                storage_manager=SimpleNamespace(
                    local_cpu_backend=SimpleNamespace(
                        memory_allocator=SimpleNamespace(buffer=slab)
                    )
                )
            )

            def shared_engine() -> Any:
                engine = make_engine(rank, root_mapping, size=2)
                if rank == 0:
                    allocator = TensorMemoryAllocator(slab)
                    allocator.shm_name = engine.shared_cpu_cache_name
                    allocator.pin_allocator = allocator
                    engine.storage_manager.local_cpu_backend.memory_allocator = (
                        allocator
                    )
                return engine

            # Install the shared slab before page_runtime constructs/registers
            # the real RemoteBackend. No parent process owns a storage loop.
            monkeypatch.setattr(local, "engine", shared_engine)
            pages = stack.enter_context(
                contextmanager(page_runtime.__wrapped__)(local, monkeypatch)
            )
            monkeypatch.setattr(
                torch.distributed, "all_gather_object", local.real_gather
            )
            monkeypatch.setattr(
                torch.distributed, "is_initialized", local.real_initialized
            )
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous}",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=30),
            )
            monkeypatch.setattr(
                vllm.distributed.parallel_state,
                "get_tp_group",
                lambda: SimpleNamespace(
                    world_size=2,
                    rank_in_group=rank,
                    cpu_group=torch.distributed.group.WORLD,
                ),
            )
            adapter = _make_async_adapter(pages, monkeypatch)
            engine = adapter.lmcache_engine
            backend, window = (
                adapter._layerwise_prefill_backend,
                adapter._layerwise_prefill_window,
            )
            connector, store = engine.gpu_connector, pages.store

            def broadcast(obj: Any, src: int) -> Any:
                payload = [obj]
                torch.distributed.broadcast_object_list(payload, src=src)
                return payload[0]

            monkeypatch.setattr(engine, "broadcast_object_fn", broadcast)
            cpu = engine.storage_manager.local_cpu_backend if rank == 0 else None
            if rank == 0:
                assert cpu.memory_allocator.buffer.data_ptr() == slab.data_ptr()
                assert engine.storage_manager.storage_backends[
                    "RemoteBackend"
                ].loop.is_running()
            else:
                assert engine.storage_manager is None
            assert slab.is_shared()
            end_logs, remote_calls = [], []
            log = sync_module.logger.info

            def capture_log(message: str, *args: Any, **kwargs: Any) -> None:
                rendered = message % args
                if rendered.startswith("[PREFILL_SYNC_STEP] event=end"):
                    end_logs.append(rendered)
                log(message, *args, **kwargs)

            monkeypatch.setattr(sync_module.logger, "info", capture_log)
            if rank == 0:
                required_put = engine.storage_manager.batched_put_sync_required

                def record_remote(keys: Any, objects: Any, **kwargs: Any) -> None:
                    if kwargs.get("location") == "RemoteBackend":
                        remote_calls.append(tuple(keys))
                    required_put(keys, objects, **kwargs)

                monkeypatch.setattr(
                    engine.storage_manager, "batched_put_sync_required", record_remote
                )

            def model_hcom() -> None:
                value = torch.tensor([rank + 1], dtype=torch.int32)
                torch.distributed.all_reduce(value)
                assert value.item() == 3

            def complete_step(req: Any) -> tuple:
                callbacks = _start(adapter, [req])
                assert len(callbacks) == 101
                assert all(
                    backend._callbacks[m.row.kv_group, m.row.row_ordinal] is m
                    for m in callbacks
                )
                before_puts, before_jobs, before_logs = (
                    len(store.puts),
                    len(remote_calls),
                    len(end_logs),
                )
                for execution in range(79):
                    current = [
                        m for m in callbacks if m.row.execution_ordinal == execution
                    ]
                    for metadata in current:
                        adapter.wait_for_layerwise_prefill_load(metadata)
                    for metadata in current:
                        _compute(engine, [req], metadata)
                        adapter.submit_layerwise_prefill_save(
                            metadata, adapter.kv_caches[metadata.row.layer_name]
                        )
                    adapter.submit_layerwise_prefill_load(current[0])
                    model_hcom()
                    for metadata in current:
                        adapter.finish_layerwise_prefill_save(metadata)
                    assert (
                        window.pending_jobs()
                        == window.pending_bytes()
                        == window.pending_futures()
                        == 0
                    )
                assert not backend._step_future.done()
                assert not adapter.layerwise_prefill_request_persist_done(
                    req.request_id
                )
                assert (
                    len(store.puts) == before_puts and len(remote_calls) == before_jobs
                )
                adapter.wait_for_save()
                assert backend._step_future.result() is None
                assert adapter.layerwise_prefill_request_persist_done(req.request_id)
                assert all(t.complete for t in connector.tickets if t.submitted)
                if rank == 0:
                    assert len(end_logs) == before_logs + 1
                    _assert_window_end_log(
                        end_logs[-1],
                        backend.window_stats(),
                        len(remote_calls) - before_jobs,
                    )
                    _assert_remote(engine, req, store)
                else:
                    assert not end_logs and not remote_calls
                return callbacks

            seed = _request(request_id="seed")
            complete_step(seed)
            adapter.get_finished({"seed"})
            torch.distributed.barrier()
            if rank == 0:
                for key in list(cpu.hot_cache):
                    cpu.remove(key, force=True)
                assert cpu.memory_allocator.num_active_allocations == 0
            torch.distributed.barrier()

            # First actual target step must reconstruct pages/tails remotely.
            cold = _request(start=300, end=530)
            old_callbacks = complete_step(cold)
            if rank == 0:
                assert len(store.gets) == 4  # Page plus legacy tail for each group.

            def forbidden_get(*args: Any, **kwargs: Any) -> None:
                raise AssertionError(
                    "Warm step queried storage instead of retained prefixes"
                )

            monkeypatch.setattr(
                engine, "resolve_layerwise_prefill_group", forbidden_get
            )
            if rank == 0:
                monkeypatch.setattr(
                    engine.storage_manager, "batched_get", forbidden_get
                )
            warm = _request(start=530, end=560, generation=2)
            if not fail:
                complete_step(warm)
                assert all(
                    old is retained
                    for old, retained in zip(
                        old_callbacks, backend._previous_callbacks, strict=True
                    )
                )
                adapter.get_finished({warm.request_id})
            else:
                callbacks = _start(adapter, [warm])
                assert all(
                    backend._callbacks[m.row.kv_group, m.row.row_ordinal] is m
                    for m in callbacks
                )
                current = callbacks[:2]
                for metadata in current:
                    adapter.wait_for_layerwise_prefill_load(metadata)
                    _compute(engine, [warm], metadata)
                    adapter.submit_layerwise_prefill_save(
                        metadata, adapter.kv_caches[metadata.row.layer_name]
                    )
                adapter.submit_layerwise_prefill_load(current[0])
                assert any(
                    t.submitted and not t.complete and not t.kwargs["direction"]
                    for t in connector.tickets
                )
                model_hcom()
                acknowledgements = []
                acknowledge = engine.layerwise_prefill_ack

                def ack(identity: Any, error: Any = None) -> None:
                    acknowledgements.append(identity)
                    acknowledge(identity, error)

                monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
                if rank == 1:
                    drain = connector.drain_layerwise_prefill_transfers

                    def delayed_drain() -> None:
                        draining.set()
                        assert allow_drain.wait(30)
                        drain()
                        assert all(t.complete for t in connector.tickets if t.submitted)
                        source_done.set()

                    monkeypatch.setattr(
                        connector, "drain_layerwise_prefill_transfers", delayed_drain
                    )
                release = backend._release

                def release_after_peer_drain(objects: list) -> None:
                    assert source_done.is_set(), (
                        "Released shared owners before peer lookahead fenced"
                    )
                    assert backend._abort_drained
                    assert all(t.complete for t in connector.tickets if t.submitted)
                    released.set()
                    release(objects)

                monkeypatch.setattr(backend, "_release", release_after_peer_drain)
                with pytest.raises(ValueError, match="identity mismatch"):
                    if rank == 0:
                        adapter.abort_layerwise_prefill_step()
                    else:
                        adapter.finish_layerwise_prefill_save(current[0])
                assert backend._step_future.exception() is not None
                adapter.abort_layerwise_prefill_step()
                window.release_request(warm.request_id)
                assert len(acknowledgements) == 2
                assert acknowledgements[1] == ("failed_step_drained", 3)
                assert not backend._bound and not backend._rows
                assert not adapter.layerwise_prefill_request_persist_done(
                    warm.request_id
                )
                assert len(end_logs) == (2 if rank == 0 else 0)
                assert backend.window_stats()["remote_jobs"] == 0

            assert not window.has_request(warm.request_id)
            assert not backend._prefixes and not backend._pending_rows
            assert (
                window.pending_jobs()
                == window.pending_bytes()
                == window.pending_futures()
                == 0
            )
            torch.distributed.barrier()
            if rank == 0:
                assert len(store.gets) == 4
                for key in list(cpu.hot_cache):
                    cpu.remove(key, force=True)
                assert cpu.memory_allocator.num_active_allocations == 0
            calls = connector.calls
            counts = (sum(c[0] for c in calls), sum(not c[0] for c in calls))
            expected_stores = (204 if fail else 303) if rank == 0 else 0
            assert counts == (expected_stores, 105 if fail else 202)
            outcome = (
                rank,
                os.getpid(),
                "drained" if fail else "roundtrip",
                counts,
                len(end_logs),
            )
        # Fixture teardown closes child-local event loops before reporting success.
        results.put(outcome)
    except BaseException:
        results.put((rank, os.getpid(), "error", traceback.format_exc()))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True], ids=["cold-warm", "abort-finish-drain"])
def test_real_gloo_async_factory_adapter_shared_slab(
    tmp_path: Any, monkeypatch: Any, fail: bool
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    context = multiprocessing.get_context("spawn")
    slab = torch.empty(8 << 20, dtype=torch.uint8).share_memory_()
    results = context.Queue()
    draining, allow_drain, source_done, released = (context.Event() for _ in range(4))
    processes = [
        context.Process(
            target=_gloo_async_worker,
            args=(
                rank,
                slab,
                str(tmp_path / "async-gloo"),
                fail,
                draining,
                allow_drain,
                source_done,
                released,
                results,
            ),
        )
        for rank in (0, 1)
    ]
    try:
        for process in processes:
            process.start()
        if fail:
            assert draining.wait(60), (
                "Peer did not enter the common failure-drain handshake"
            )
            assert not source_done.is_set() and not released.is_set()
            allow_drain.set()
        outcomes = [results.get(timeout=90) for _ in processes]
        assert all(outcome[2] != "error" for outcome in outcomes), outcomes
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert {outcome[1] for outcome in outcomes} == {p.pid for p in processes}
        assert os.getpid() not in {outcome[1] for outcome in outcomes}
        assert sorted((o[0], o[2], o[4]) for o in outcomes) == [
            (0, "drained" if fail else "roundtrip", 2 if fail else 3),
            (1, "drained" if fail else "roundtrip", 0),
        ]
        if fail:
            assert source_done.is_set() and released.is_set()
    finally:
        allow_drain.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
        results.close()
        results.join_thread()
