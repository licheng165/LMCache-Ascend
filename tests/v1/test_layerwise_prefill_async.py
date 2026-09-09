# SPDX-License-Identifier: Apache-2.0
"""Async row ownership with real CPU slabs, required queue and page-only remote."""

# ruff: noqa: F811

# Standard
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import pickle

# Third Party
from lmcache.v1.storage_backend.required_put_queue import RequiredPutQueue
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    LayerwisePrefillCallbackMetadata,
)
from vllm.v1.kv_cache_interface import DSAExecutionRow, DSAKVRow
import pytest
import torch

# First Party
from lmcache_ascend.v1.layerwise_prefill_async import LayerwisePrefillAsyncBackend
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillFenceError
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)
import lmcache_ascend.v1.layerwise_prefill_async as async_module
import lmcache_ascend.v1.layerwise_prefill_sync as sync_module

# Local
from tests.v1.test_layerwise_prefill_row_transfer import (
    _NPUTensor,
    row_env,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync import (
    CPUConnector,
    _metadata,
    _registry,
    _request,
    _sentinel,
    _slots,
    _view,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_pages import (
    _assert_remote,
    page_runtime,  # noqa: F401
)


class DeferredCPUConnector(CPUConnector):
    """Deferred bank copies; no payload is touched by prepare/submit."""

    supports_layerwise_prefill_async_rows = True

    def __init__(self, blocks: int = 64) -> None:
        super().__init__(blocks)
        self.tickets = []
        self.events = []
        self.gate = Event()
        self.gate.set()
        self.completing = Event()
        self.launch_error = False
        self.fence_error = False

    def prepare_layerwise_prefill_row(self, *args: Any, **kwargs: Any) -> Any:
        self.events.append(("prepare", kwargs["direction"], kwargs["kv_group"]))
        ticket = SimpleNamespace(
            args=args,
            kwargs=kwargs,
            submitted=False,
            complete=False,
            dependency=None,
        )
        ticket.done_event = ticket
        self.tickets.append(ticket)
        return ticket

    def submit_layerwise_prefill_row(
        self, ticket: Any, *, wait_event: Any = None
    ) -> Any:
        assert not ticket.submitted
        ticket.submitted = True
        ticket.dependency = wait_event
        self.events.append(
            (
                "submit",
                ticket.kwargs["direction"],
                ticket.kwargs["kv_group"],
                wait_event,
            )
        )
        if self.launch_error:
            raise RuntimeError("partial native launch failed")
        return ticket

    def _run(self, ticket: Any) -> None:
        if ticket.complete:
            return
        assert ticket.submitted
        if ticket.dependency is not None:
            self._run(ticket.dependency)
        if ticket.args[1]:
            super().transfer_layerwise_prefill_row(*ticket.args, **ticket.kwargs)
        ticket.complete = True
        ticket.args = ()

    def wait_layerwise_prefill_row(self, ticket: Any) -> None:
        self.events.append(("compute_wait",))
        self._run(ticket)

    def complete_layerwise_prefill_row(self, ticket: Any) -> None:
        self.events.append(("host_fence",))
        self.completing.set()
        assert self.gate.wait(10)
        if self.fence_error:
            raise RuntimeError("unknown completion fence")
        self._run(ticket)

    def drain_layerwise_prefill_transfers(self) -> None:
        self.events.append(("drain",))
        for ticket in self.tickets:
            if ticket.submitted and not ticket.complete:
                self.complete_layerwise_prefill_row(ticket)


def _backend(engine: Any) -> LayerwisePrefillAsyncBackend:
    previous = engine.gpu_connector
    engine.gpu_connector = DeferredCPUConnector(previous.planes[0][0].shape[0])
    backend = LayerwisePrefillAsyncBackend(engine, _view())
    engine._layerwise_prefill_window_backend = backend
    return backend


def _callbacks(requests: list) -> tuple:
    generations = tuple((r.request_id, r.allocation_generation) for r in requests)
    callbacks = []
    for execution, latent, indexer in _view().executions:
        entry = DSAExecutionRow(
            execution,
            DSAKVRow(*latent),
            DSAKVRow(*indexer) if indexer is not None else None,
        )
        for row in (entry.latent, entry.indexer):
            if row is not None:
                callbacks.append(
                    LayerwisePrefillCallbackMetadata(entry, row, generations)
                )
    return tuple(callbacks)


def _bind(
    engine: Any, backend: Any, requests: list, callbacks: tuple | None = None
) -> tuple:
    callbacks = _callbacks(requests) if callbacks is None else callbacks
    backend.bind_step(requests, _registry(engine), callbacks=callbacks)
    return callbacks


def _compute(engine: Any, requests: list, metadata: Any) -> None:
    group, row, bank = (
        metadata.row.kv_group,
        metadata.row.row_ordinal,
        metadata.row.bank,
    )
    for req in requests:
        slots = _slots(req, bank, group, req.compute_end)
        for plane_index, plane in enumerate(_registry(engine)[metadata.row.layer_name]):
            expected = _sentinel(req, group, row, plane_index, req.compute_end)
            assert torch.equal(
                plane.view(-1)[slots[: req.restore_end]], expected[: req.restore_end]
            )
            plane.view(-1)[slots[req.compute_start :]] = expected[req.compute_start :]


def _execute(
    engine: Any,
    backend: Any,
    requests: list,
    *,
    finish: bool = True,
    env: Any = None,
    callbacks: tuple | None = None,
) -> Any:
    registry = _registry(engine)
    all_callbacks = _bind(engine, backend, requests, callbacks)
    future = None
    for execution, latent, indexer in _view().executions:
        callbacks = [m for m in all_callbacks if m.row.execution_ordinal == execution]
        for metadata in callbacks:
            before = len(env.events) if env is not None else 0
            backend.wait_for_load(metadata)
            if env is not None:
                assert not any("sync" in str(e[1]) for e in env.events[before:])
        for metadata in callbacks:
            if env is None:
                _compute(engine, requests, metadata)
            else:
                env.compute.queue.append(
                    lambda metadata=metadata: _compute(engine, requests, metadata)
                )
            backend.submit_save(metadata, registry[metadata.row.layer_name])
        backend.submit_load(callbacks[0])
        if env is not None:
            env.compute.run_to(len(env.compute.queue))
        for metadata in callbacks:
            returned = backend.finish_save(metadata)
            assert future is None or returned is future
            future = returned
            assert not future.done()
        assert (
            backend.pending_jobs()
            == backend.pending_bytes()
            == backend.pending_futures()
            == 0
        )
    if finish:
        backend.finish_step()
        assert future.done() and future.result() is None
    return future


@pytest.mark.parametrize("error_type", [None, ValueError, KeyboardInterrupt])
def test_timed_call_preserves_result_error_and_exact_host_time(
    monkeypatch: Any, error_type: type | None
) -> None:
    backend = object.__new__(LayerwisePrefillAsyncBackend)
    backend._host_timings = {}
    clock = Mock(side_effect=[10.0, 10.125, 100.0, 100.5, 200.0, 200.25])
    monkeypatch.setattr(async_module, "perf_counter", clock)
    argument, result = object(), object()
    function = Mock(return_value=result)
    assert backend._timed_call("prepare_load", function, argument, flag=True) is result
    function.assert_called_once_with(argument, flag=True)

    error = error_type("timed sentinel") if error_type is not None else None
    function = Mock(return_value=result, side_effect=error)
    if error is None:
        assert backend._timed_call("prepare_load", function, argument) is result
    else:
        with pytest.raises(error_type) as caught:
            backend._timed_call("prepare_load", function, argument)
        assert caught.value is error
    function.assert_called_once_with(argument)
    assert backend._timed_call("submit_load", lambda: result) is result
    assert backend._host_timings == {
        "prepare_load": (2, 0.625, 0.5),
        "submit_load": (1, 0.25, 0.25),
    }
    assert clock.call_count == 6


def test_async_host_timing_scopes_include_save_preparation_and_publication_ack(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    connector = engine.gpu_connector
    clock = [0.0]
    monkeypatch.setattr(async_module, "perf_counter", lambda: clock[0])

    def delay(owner: Any, name: str, seconds: float) -> None:
        function = getattr(owner, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            finally:
                clock[0] += seconds

        monkeypatch.setattr(owner, name, timed)

    # Only model-thread operations advance this clock, never remote workers.
    for owner, name, seconds in (
        (backend, "_plan", 0.002),
        (engine, "layerwise_prefill_row_metadata", 0.004),
        (engine.storage_manager, "allocate", 0.008),
        (engine, "resolve_layerwise_prefill_row", 0.016),
        (engine, "layerwise_prefill_ack", 0.032),
        (connector, "prepare_layerwise_prefill_row", 0.001),
        (connector, "submit_layerwise_prefill_row", 0.002),
        (connector, "wait_layerwise_prefill_row", 0.004),
        (connector, "complete_layerwise_prefill_row", 0.008),
        (connector, "drain_layerwise_prefill_transfers", 0.016),
        (connector, "synchronize_dense_load_stream", 0.032),
        (connector, "synchronize_shared_cpu_store_publication", 0.064),
        (RequiredPutQueue, "submit", 0.128),
        (RequiredPutQueue, "close", 0.256),
    ):
        delay(owner, name, seconds)

    _execute(engine, backend, [_request()])
    assert backend.window_stats()["async_host"] == {
        "window_bind": (1, 32.0, 32.0),
        # H2D excludes source resolution, planning and preparation ACKs.
        "prepare_load": (101, 101.0, 1.0),
        # One scope per row: plan + two metadata/allocations + connector prepare.
        "prepare_save": (101, 2727.0, 27.0),
        "submit_load": (101, 202.0, 2.0),
        "submit_save": (101, 202.0, 2.0),
        "wait_load": (101, 404.0, 4.0),
        "complete_load": (101, 808.0, 8.0),
        "complete_save": (101, 808.0, 8.0),
        # Publication includes resolve + save ACK, but not source_done ACK.
        "publish": (101, 4848.0, 48.0),
        "drain": (1, 112.0, 112.0),
        "remote_submit": (2, 256.0, 128.0),
        "remote_drain": (1, 256.0, 256.0),
    }


@pytest.mark.parametrize("request_count", [1, 2])
def test_async_metrics_reset_at_bind_root_passive_and_log_snapshots(
    page_runtime: Any, monkeypatch: Any, request_count: int
) -> None:
    engines, backends = _tp_backends(page_runtime)
    resets = []
    stream_syncs = []
    for engine, backend in zip(engines, backends, strict=True):
        reset = Mock(wraps=engine.reset_layerwise_prefill_ack_stats)
        monkeypatch.setattr(engine, "reset_layerwise_prefill_ack_stats", reset)
        resets.append(reset)
        acknowledge = engine.layerwise_prefill_ack

        def ack(
            identity: Any,
            error: Any = None,
            *,
            engine: Any = engine,
            backend: Any = backend,
            acknowledge: Any = acknowledge,
        ) -> None:
            if isinstance(identity[1], tuple) and identity[1][0] == "window_bind":
                assert backend._host_timings == {}
                assert engine.layerwise_prefill_ack_stats()["count"] == 0
            acknowledge(identity, error)

        monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
        for name in (
            "synchronize_dense_load_stream",
            "synchronize_shared_cpu_store_publication",
        ):
            sync = Mock(wraps=getattr(engine.gpu_connector, name))
            monkeypatch.setattr(engine.gpu_connector, name, sync)
            stream_syncs.append(sync)

    for device in (torch.npu, torch.cuda):
        for name in ("Event", "synchronize"):
            monkeypatch.setattr(
                device, name, Mock(side_effect=AssertionError("diagnostic device work"))
            )
    tensor_gather = Mock(wraps=torch.distributed.all_gather)
    object_gather = Mock(wraps=torch.distributed.all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_gather", tensor_gather)
    monkeypatch.setattr(torch.distributed, "all_gather_object", object_gather)
    logged = []
    logger = Mock()
    logger.info.side_effect = lambda message, *args: logged.append(
        (page_runtime.thread.rank, message % args)
    )
    monkeypatch.setattr(sync_module, "logger", logger)
    snapshots, expected_snapshots = [], []
    slow_calls = 0
    ack_counts = {
        "window_bind": 1,
        "bind": 1,
        "validate": 101,
        "prepare_load": 101 * request_count,
        "load_prepared": 101 * request_count,
        "prepare_save": 101,
        "load_ready": 101,
        "source_done": 101,
        "save": 101 * request_count,
        "device_finish": 1,
        "finish": 1,
        "commit": 1,
    }
    ack_calls = 101 * (4 + 3 * request_count) + 5
    assert sum(ack_counts.values()) == ack_calls
    assert request_count != 1 or ack_calls == 712

    for step, (start, end) in enumerate(((0, 300), (300, 530)), start=1):
        requests = [
            _request(i, start=start, end=end, generation=step)
            for i in range(request_count)
        ]
        page_runtime.parallel(
            *[
                lambda rank=rank, requests=requests: _execute(
                    engines[rank], backends[rank], requests
                )
                for rank in (0, 1)
            ]
        )
        assert snapshots == expected_snapshots
        for rank, (engine, backend) in enumerate(zip(engines, backends, strict=True)):
            stats = backend.window_stats()
            load_count = 101 * request_count
            save_count = load_count if rank == 0 else 101
            host_counts = {
                "window_bind": 1,
                "prepare_load": load_count,
                "prepare_save": 101,
                "submit_load": load_count,
                "submit_save": save_count,
                "wait_load": load_count,
                "complete_load": load_count,
                "complete_save": save_count,
                "publish": 101,
                "drain": 1,
            }
            if rank == 0:
                host_counts.update(remote_submit=2 * request_count, remote_drain=1)
            assert {key: value[0] for key, value in stats["async_host"].items()} == (
                host_counts
            )
            for calls, total_ms, max_ms in stats["async_host"].values():
                assert calls > 0 and total_ms >= max_ms >= 0

            ack_stats = engine.layerwise_prefill_ack_stats()
            assert stats["ack"] == {
                "calls": ack_calls,
                "ms": round(ack_stats["total_ms"], 3),
                "max_ms": round(ack_stats["max_ms"], 3),
                "fast": ack_stats["fast_count"],
                "slow": ack_stats["slow_count"],
                "payload_bytes": ack_stats["serialized_bytes"],
                "max_payload_bytes": ack_stats["max_payload_bytes"],
                "phase": {
                    phase: (
                        values["count"],
                        round(values["total_ms"], 3),
                        round(values["max_ms"], 3),
                    )
                    for phase, values in ack_stats["by_phase"].items()
                },
            }
            assert {
                phase: values[0] for phase, values in stats["ack"]["phase"].items()
            } == ack_counts
            assert ack_stats["count"] == ack_calls
            assert ack_stats["fast_count"] + ack_stats["slow_count"] == ack_calls
            assert ack_stats["serialized_bytes"] >= ack_stats["max_payload_bytes"] > 0
            slow_calls += ack_stats["slow_count"]
            assert resets[rank].call_count == step
            assert Counter(event[0] for event in engine.gpu_connector.events) == {
                "prepare": (load_count + save_count) * step,
                "submit": (load_count + save_count) * step,
                "compute_wait": load_count * step,
                "host_fence": (load_count + save_count) * step,
                "drain": step,
            }

            snapshots.append(stats)
            expected_snapshots.append(deepcopy(stats))
            changed = backend.window_stats()
            changed["async_host"].clear()
            changed["ack"]["phase"].clear()
            changed["ack"]["calls"] = -1
            assert backend.window_stats() == stats
        assert tensor_gather.call_count == 2 * ack_calls * step
        assert object_gather.call_count == slow_calls
        assert all(sync.call_count == step for sync in stream_syncs)
        assert len(logger.method_calls) == len(logged) == 2 * step
        assert all(rank == 0 for rank, _ in logged)
        begin, end_line = [line for _, line in logged[-2:]]
        assert f"event=begin step={step} requests={request_count}" in begin
        assert "async_host" not in begin and "window_stats" not in begin
        assert (
            f"event=end step={step} requests={request_count} saved=(79, 22)" in end_line
        )
        assert "transfer_timing=False" in end_line
        assert ast.literal_eval(end_line.split(" window_stats=", 1)[1]) == (
            backends[0].window_stats()
        )


@pytest.mark.parametrize("request_count", [1, 4])
def test_full_two_steps_page_only_and_warm_manifest(
    page_runtime: Any, monkeypatch: Any, request_count: int
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    requests = [_request(i) for i in range(request_count)]
    future = _execute(engine, backend, requests, finish=False)
    assert not page_runtime.store.puts
    assert backend.pending_futures() == 0 and not future.cancel()
    backend.finish_step()
    for req in requests:
        _assert_remote(engine, req, page_runtime.store)
    monkeypatch.setattr(
        engine,
        "resolve_layerwise_prefill_group",
        Mock(side_effect=AssertionError("warm group lookup")),
    )
    requests = [
        replace(
            req,
            compute_start=300,
            restore_end=300,
            compute_end=530,
            allocation_generation=2,
        )
        for req in requests
    ]
    _execute(engine, backend, requests)
    for req in requests:
        _assert_remote(engine, req, page_runtime.store)
    stats = backend.window_stats()
    assert stats["max_jobs"] == stats["max_futures"] == 8
    assert stats["max_bytes"] == 64 << 20
    assert stats["device_jobs"] == 101
    assert stats["remote_jobs"] == 2 * request_count
    assert stats["peak_jobs"] <= 8 and stats["peak_futures"] <= 8


def test_root_passive_two_steps_and_group_bank_golden(page_runtime: Any) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backends.append(_backend(engine))
    for start, end in ((0, 300), (300, 530)):
        requests = [_request(start=start, end=end, generation=1 + bool(start))]
        page_runtime.parallel(
            *[
                lambda rank=rank, requests=requests: _execute(
                    engines[rank], backends[rank], requests
                )
                for rank in (0, 1)
            ]
        )
    assert not any(call[0] for call in engines[1].gpu_connector.calls)
    assert len(engines[1].gpu_connector.calls) == 101
    # INDEXER execution 6 is row 3/bank 1, execution 78 is row 21/bank 1.
    assert _view().executions[6][2][3:] == (3, 1)
    assert _view().executions[78][2][3:] == (21, 1)
    for engine in engines:
        tickets = engine.gpu_connector.tickets
        loads = [t for t in tickets if not t.kwargs["direction"]]
        assert len(loads) == 202
        for group, count in enumerate((79, 22)):
            group_loads = [t for t in loads[101:] if t.kwargs["kv_group"] == group]
            assert len(group_loads) == count
            assert all(t.dependency is not None for t in group_loads[2:])
            assert all(
                t.dependency.kwargs["kv_group"] == group for t in group_loads[2:]
            )
            group_stores = [
                t
                for t in tickets
                if t.kwargs["direction"] and t.kwargs["kv_group"] == group
            ][count:]
            assert all(
                load.dependency is group_stores[row - 2]
                for row, load in enumerate(group_loads)
                if row >= 2
            )


def test_cold_page_only_partial_bootstrap_once_per_group(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _execute(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    cpu = engine.storage_manager.local_cpu_backend
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)
    resolve = Mock(wraps=engine.resolve_layerwise_prefill_group)
    monkeypatch.setattr(engine, "resolve_layerwise_prefill_group", resolve)
    req = _request(start=300, end=530, request_id="external")
    _execute(engine, backend, [req])
    assert resolve.call_count == 2
    assert page_runtime.store.gets
    _assert_remote(engine, req, page_runtime.store)


def test_pre_hcom_only_enqueues_and_delayed_d2h_not_published(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    requests, registry = [_request()], _registry(engine)
    callbacks = _bind(engine, backend, requests)[:2]
    for metadata in callbacks:
        backend.wait_for_load(metadata)
    connector = engine.gpu_connector
    before = len(connector.events)
    with monkeypatch.context() as hot:
        for owner, name in (
            (engine, "layerwise_prefill_ack"),
            (engine, "resolve_layerwise_prefill_row"),
            (engine.storage_manager, "allocate"),
            (engine.storage_manager, "batched_put_sync_required"),
            (connector, "prepare_layerwise_prefill_row"),
            (connector, "complete_layerwise_prefill_row"),
        ):
            hot.setattr(
                owner, name, Mock(side_effect=AssertionError("host work pre-HCOM"))
            )
        for metadata in callbacks:
            backend.submit_save(metadata, registry[metadata.row.layer_name])
        backend.submit_load(callbacks[0])
    assert all(event[0] == "submit" for event in connector.events[before:])
    assert not connector.calls
    assert not engine.storage_manager.local_cpu_backend.hot_cache
    assert backend.pending_jobs() == 2
    assert backend.pending_bytes() == 1800
    assert backend.pending_futures() == 0
    connector.gate.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        completing = pool.submit(backend.finish_save, callbacks[0])
        assert connector.completing.wait(5)
        assert not completing.done()
        assert not engine.storage_manager.local_cpu_backend.hot_cache
        assert not page_runtime.store.puts
        connector.gate.set()
        future = completing.result(5)
    assert not future.done()
    assert backend.pending_jobs() == 1 and backend.pending_bytes() == 600
    backend.finish_save(callbacks[1])
    backend.abort_request("req-0")


@pytest.mark.parametrize(
    "limits", [(True, 1 << 26, 8), (8, 1, 8), (1, 1 << 26, 8), (8, 1 << 26, 0)]
)
def test_limits_rejected_all_ranks_before_io(page_runtime: Any, limits: tuple) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backend = _backend(engine)
        backend.configure_window_limits(*limits)
        backends.append(backend)

    def bind(rank: int) -> None:
        with pytest.raises(ValueError, match="limits|page|budget"):
            _bind(engines[rank], backends[rank], [_request()])

    page_runtime.parallel(lambda: bind(0), lambda: bind(1))
    assert not page_runtime.store.puts and not page_runtime.store.gets
    assert all(not engine.gpu_connector.prepared for engine in engines)


@pytest.mark.parametrize(
    "failure", ["submit", "identity", "finish_identity", "duplicate"]
)
def test_errors_deferred_until_post_hcom_all_tp(
    page_runtime: Any, failure: str
) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backends.append(_backend(engine))
    hcom = Event()

    def run(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        metadata = _bind(engine, backend, [_request()])[0]
        backend.wait_for_load(metadata)
        if rank == 0:
            engine.gpu_connector.launch_error = failure == "submit"
        error = (
            ValueError("coordinator identity")
            if rank == 0 and failure == "identity"
            else None
        )
        backend.submit_save(
            metadata, _registry(engine)[metadata.row.layer_name], validation_error=error
        )
        if rank == 0 and failure == "duplicate":
            backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
        hcom.set()  # Reached model HCOM despite local enqueue/identity failure.
        error = (
            ValueError("coordinator finish identity")
            if rank == 0 and failure == "finish_identity"
            else None
        )
        with pytest.raises(ValueError, match="native|identity|duplicated"):
            backend.finish_save(metadata, validation_error=error)
        backend.abort_request("req-0")
        with pytest.raises(ValueError, match="poisoned"):
            _bind(engine, backend, [_request(generation=2)])

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    assert hcom.is_set() and not page_runtime.store.puts


def test_byte_batched_commit_and_slow_storage_bounded(page_runtime: Any) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    backend.configure_window_limits(8, 79 * 256 * 4, 1)
    future = _execute(engine, backend, [_request(end=530)], finish=False)
    store = page_runtime.store
    store.gate.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        finished = pool.submit(backend.finish_step)
        assert store.submitted.wait(5)
        assert not finished.done() and not future.done()
        assert backend.pending_jobs() == backend.pending_futures() == 1
        assert backend.pending_bytes() <= 79 * 256 * 4
        store.gate.set()
        finished.result(10)
    assert future.result() is None
    stats = backend.window_stats()
    assert stats["remote_jobs"] == 4  # Three LATENT batches, one INDEXER batch.
    assert stats["peak_futures"] == 1 and stats["peak_bytes"] <= stats["max_bytes"]


def _real_backend(engine: Any, env: Any, monkeypatch: Any) -> Any:
    connector = env.connector
    connector.planes = {
        group: tuple(plane.as_subclass(_NPUTensor) for plane in planes)
        for group, planes in engine.gpu_connector.planes.items()
    }
    connector._group_layouts.clear()
    connector.lmcache_chunk_size = 256
    # row_env forbids global initialization inside row hooks. Bind legitimately
    # initializes it once; restore those real methods for this integration test.
    for name in ("initialize_kvcaches_ptr", "_lazy_initialize_buffer"):
        monkeypatch.setattr(
            connector,
            name,
            getattr(VLLMPagedMemLayerwiseNPUConnector, name).__get__(connector),
        )
    engine.gpu_connector = connector
    backend = LayerwisePrefillAsyncBackend(engine, _view())
    engine._layerwise_prefill_window_backend = backend
    return backend


def test_real_connector_two_steps_no_pending_native_work(
    page_runtime: Any, row_env: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _real_backend(engine, row_env, monkeypatch)
    for start, end in ((0, 300), (300, 530)):
        requests = [_request(start=start, end=end, generation=1 + bool(start))]
        _execute(engine, backend, requests, env=row_env)
        assert not row_env.pending
        assert row_env.store.completed == len(row_env.store.queue)
        assert row_env.load.completed == len(row_env.load.queue)
        # Only the remote byte assertion needs ordinary CPU concatenation.
        with monkeypatch.context() as check:
            check.setattr(torch, "cat", torch.concat)
            _assert_remote(engine, requests[0], page_runtime.store)


@pytest.mark.parametrize("failure", [None, "native", "event"])
def test_real_connector_launch_before_projection_fence_after_hcom(
    page_runtime: Any, row_env: Any, monkeypatch: Any, failure: str | None
) -> None:
    engine = page_runtime.engine()
    backend = _real_backend(engine, row_env, monkeypatch)
    requests = [_request()]
    registry = _registry(engine)
    metadata = _bind(engine, backend, requests)[0]
    backend.wait_for_load(metadata)
    row_env.compute.queue.append(lambda: _compute(engine, requests, metadata))
    before = len(row_env.events)
    host_before = dict(row_env.host_ops)
    row_env.launch_error = failure == "native"
    row_env.event_record_error = failure == "event"
    backend.submit_save(metadata, registry[metadata.row.layer_name])
    assert row_env.pending
    assert ("store", "launch") in row_env.events[before:]
    assert not any("sync" in str(event[1]) for event in row_env.events[before:])
    assert row_env.host_ops == host_before
    assert not engine.storage_manager.local_cpu_backend.hot_cache
    row_env.events.append(("model", "projection"))
    row_env.compute.run_to(len(row_env.compute.queue))
    row_env.events.append(("model", "hcom"))
    if failure:
        with pytest.raises(ValueError, match="native launch|event record"):
            backend.finish_save(metadata)
    else:
        assert not backend.finish_save(metadata).done()
    assert not row_env.pending
    assert row_env.events.index(("model", "hcom")) < next(
        i
        for i in range(before, len(row_env.events))
        if "sync" in str(row_env.events[i][1])
    )
    backend.abort_request("req-0")


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_real_prepare_failure_or_abort_before_finish_releases_only_after_drain(
    page_runtime: Any, row_env: Any, monkeypatch: Any, fail_prepare: bool
) -> None:
    engine = page_runtime.engine()
    backend = _real_backend(engine, row_env, monkeypatch)
    requests = [_request()]
    registry = _registry(engine)
    metadata = _bind(engine, backend, requests)[0]
    if fail_prepare:
        row_env.pointer_error = "raise"
        with pytest.raises(ValueError, match="registration lookup"):
            backend.wait_for_load(metadata)
        assert not row_env.calls
    else:
        backend.wait_for_load(metadata)
        row_env.compute.queue.append(lambda: _compute(engine, requests, metadata))
        backend.submit_save(metadata, registry[metadata.row.layer_name])
        assert row_env.pending
    allocator = engine.storage_manager.local_cpu_backend.memory_allocator
    assert allocator.num_active_allocations > 0
    backend.abort_request("req-0")
    assert not row_env.pending
    assert allocator.num_active_allocations == 0
    with pytest.raises(ValueError, match="poisoned"):
        _bind(engine, backend, [_request(generation=2)])


@pytest.mark.parametrize("at_abort", [False, True])
def test_unknown_fence_quarantines_every_rank_and_all_step_owners(
    page_runtime: Any, at_abort: bool
) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backends.append(_backend(engine))
    page_runtime.parallel(
        lambda: _execute(engines[0], backends[0], [_request()]),
        lambda: _execute(engines[1], backends[1], [_request()]),
    )
    requests = [_request(start=300, end=530, generation=2)]

    def run(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        metadata = _bind(engine, backend, requests)[0]
        backend.wait_for_load(metadata)
        backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
        engine.gpu_connector.fence_error = rank == 1
        if not at_abort:
            with pytest.raises(LayerwisePrefillFenceError, match="unknown"):
                backend.finish_save(metadata)
        with pytest.raises(LayerwisePrefillFenceError, match="Cannot release"):
            backend.abort_request("req-0")
        assert backend._unsafe_transfer
        assert backend._prefixes and backend._rows
        assert all(
            obj.is_valid() and obj.metadata.pin_count == 1
            for prefix in backend._prefixes.values()
            for obj in prefix.objects
        )

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    allocator = engines[0].storage_manager.local_cpu_backend.memory_allocator
    assert allocator.num_active_allocations == 204
    for source in backends[0]._rows[0, 0].sources:
        assert all(
            obj.is_valid() and obj.metadata.pin_count == 1 for obj in source.fresh
        )
    # CPU-only teardown simulates process exit; production has no recovery API.
    for engine, backend in zip(engines, backends, strict=True):
        engine.gpu_connector.fence_error = False
        engine.gpu_connector.drain_layerwise_prefill_transfers()
        backend._unsafe_transfer = False
        backend._abort_drained = True
        backend.abort_request("req-0")


def test_stale_generation_finish_preserves_current_record(page_runtime: Any) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    callbacks = _callbacks([_request()])
    _execute(engine, backend, [_request()], callbacks=callbacks)
    old = callbacks[0]
    req = _request(start=300, end=530, generation=2)
    metadata = _bind(engine, backend, [req])[0]
    backend.wait_for_load(metadata)
    _compute(engine, [req], metadata)
    backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
    current = backend._rows[0, 0]
    with pytest.raises(ValueError, match="generations"):
        backend.finish_save(old)
    assert backend._rows[0, 0] is current and not current.finished
    assert current.sources and all(obj.is_valid() for obj in current.sources[0].fresh)
    backend.abort_request("req-0")


@pytest.mark.parametrize("failure", [False, True])
def test_final_remote_ack_completes_all_tp_only_after_queue_close(
    page_runtime: Any, failure: bool
) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backends.append(_backend(engine))
    requests = [_request()]
    futures = page_runtime.parallel(
        *[
            lambda rank=rank: _execute(
                engines[rank], backends[rank], requests, finish=False
            )
            for rank in (0, 1)
        ]
    )
    page_runtime.store.fail = failure
    page_runtime.store.gate.clear()

    def finish(rank: int) -> None:
        page_runtime.thread.rank = rank
        backend = backends[rank]
        if failure:
            with pytest.raises(ValueError, match="put|failed"):
                backend.finish_step()
        else:
            backend.finish_step()
        assert (
            backend.pending_jobs()
            == backend.pending_bytes()
            == backend.pending_futures()
            == 0
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(finish, rank) for rank in (0, 1)]
        assert page_runtime.store.submitted.wait(5)
        assert all(not future.done() for future in futures)
        assert all(not result.done() for result in results)
        page_runtime.store.gate.set()
        for result in results:
            result.result(10)
    assert all(future.done() for future in futures)
    if failure:
        assert all(future.exception() is not None for future in futures)
        page_runtime.parallel(
            lambda: backends[0].abort_request("req-0"),
            lambda: backends[1].abort_request("req-0"),
        )


@pytest.mark.parametrize(
    "failure,bad_rank",
    [("plan", 0), ("plan", 1), ("prepare", 0), ("prepare", 1), ("allocate", 0)],
)
def test_row_preparation_errors_ack_all_ranks_before_sfa(
    page_runtime: Any, monkeypatch: Any, failure: str, bad_rank: int
) -> None:
    engines = [page_runtime.engine(size=2)]
    engines.append(page_runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        backends.append(_backend(engine))
    if failure == "plan":
        monkeypatch.setattr(
            backends[bad_rank],
            "_plan",
            Mock(side_effect=RuntimeError("prepare plan failed")),
        )
    elif failure == "prepare":
        monkeypatch.setattr(
            engines[bad_rank].gpu_connector,
            "prepare_layerwise_prefill_row",
            Mock(side_effect=RuntimeError("prepare native failed")),
        )
    else:
        manager = engines[0].storage_manager
        allocate = manager.allocate
        count = 0

        def partial(*args: Any, **kwargs: Any) -> Any:
            nonlocal count
            count += 1
            return allocate(*args, **kwargs) if count == 1 else None

        monkeypatch.setattr(manager, "allocate", partial)

    def run(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        requests = [_request()]
        metadata = _bind(engine, backend, requests)[0]
        with pytest.raises(ValueError, match="prepare|allocation"):
            backend.wait_for_load(metadata)
        assert not engine.gpu_connector.calls
        backend.abort_request("req-0")

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    assert (
        engines[
            0
        ].storage_manager.local_cpu_backend.memory_allocator.num_active_allocations
        == 0
    )


def test_default_64mib_batches_two_latent_pages_by_actual_object_bytes(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    req = replace(
        _request(),
        token_ids=tuple(range(768)),
        compute_end=768,
        block_ids_by_bank=tuple(
            (tuple(range(1 + bank * 6, 7 + bank * 6)),) * 2 for bank in (0, 1)
        ),
    )
    _execute(engine, backend, [req], finish=False)
    # Exercise the actual object-size batching hook with production LATENT byte
    # widths, without allocating a production-sized 79-row slab in a CPU test.
    objects = [
        obj
        for identity, prefix in backend._prefixes.items()
        if identity[2] == 0
        for obj in prefix.objects
    ]
    with monkeypatch.context() as sizes:
        for obj in objects:
            sizes.setattr(obj, "get_size", lambda: 256 * 576 * 2)
        batches = list(backend._page_batches(req, 0, 0))
        assert batches == [range(0, 2), range(2, 3)]
    backend.finish_step()


def test_configure_limits_has_no_storage_construction_dependency(
    page_runtime: Any,
) -> None:
    engine = page_runtime.engine()
    manager = engine.storage_manager
    engine.storage_manager = None
    backend = _backend(engine)
    backend.configure_window_limits(max_jobs=8, max_bytes=64 << 20, max_futures=8)
    assert backend.window_stats()["max_bytes"] == 64 << 20
    assert backend.pending_jobs() == backend.pending_futures() == 0
    engine.storage_manager = manager


@pytest.mark.parametrize("end", [300, 3500])
def test_joint_execution_budget_counts_batches_not_requests(
    page_runtime: Any, end: int
) -> None:
    engine = page_runtime.engine(blocks=260)
    backend = _backend(engine)
    backend.configure_window_limits(2, 79 * 256 * 4, 1)
    requests = [
        replace(
            _request(i),
            token_ids=tuple(range(i * 1000, i * 1000 + end)),
            compute_end=end,
            block_ids_by_bank=tuple(
                (tuple(range(1 + i * 60 + bank * 30, 31 + i * 60 + bank * 30)),) * 2
                for bank in (0, 1)
            ),
        )
        for i in range(4)
    ]
    if end == 3500:
        # Each changed row batch fits separately; their joint execution does not.
        assert 4 * end * 4 < 79 * 256 * 4 < 4 * end * 6
        with pytest.raises(ValueError, match="Current execution row batches"):
            _bind(engine, backend, requests)
        assert not engine.gpu_connector.prepared
        assert not page_runtime.store.puts and not page_runtime.store.gets
    else:
        _execute(engine, backend, requests)
        assert backend.window_stats()["peak_jobs"] == 2
        assert backend.window_stats()["device_jobs"] == 101


def _tp_backends(runtime: Any) -> tuple[list, list]:
    engines = [runtime.engine(size=2)]
    engines.append(runtime.engine(1, engines[0], size=2))
    backends = []
    for rank, engine in enumerate(engines):
        runtime.thread.rank = rank
        backends.append(_backend(engine))
    return engines, backends


def _dispose_cpu_quarantine(engines: list, backends: list) -> None:
    # Simulate process exit after the CPU mock can complete. There is deliberately
    # no public production API that makes an unknown-fence backend reusable.
    for engine, backend in zip(engines, backends, strict=True):
        engine.gpu_connector.fence_error = False
        engine.gpu_connector.drain_layerwise_prefill_transfers()
        backend._unsafe_transfer = False
        backend._abort_drained = True
        backend.abort_step()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_arg",
        "missing_row",
        "different_alias",
        "row",
        "execution",
        "generations",
    ],
)
def test_callback_registry_rejected_before_bind_io_all_tp(
    page_runtime: Any, failure: str
) -> None:
    engines, backends = _tp_backends(page_runtime)
    requests = [_request()]

    def run(rank: int) -> None:
        callbacks = _callbacks(requests)
        if rank == 1:
            if failure == "missing_arg":
                callbacks = None
            elif failure == "missing_row":
                callbacks = callbacks[:-1]
            elif failure == "different_alias":
                callbacks += (replace(callbacks[0]),)
            else:
                bad = _metadata(_view(), _view().rows_by_group[0][0], requests)
                if failure == "row":
                    bad.row.layer_name = "not-a-canonical-layer"
                elif failure == "execution":
                    bad.execution.indexer = None
                else:
                    bad.request_generations = ((requests[0].request_id, 2),)
                callbacks = (bad, *callbacks[1:])
        with pytest.raises(
            ValueError, match="callbacks|alias|membership|execution|generations"
        ):
            backends[rank].bind_step(
                requests, _registry(engines[rank]), callbacks=callbacks
            )
        assert backends[rank]._abort_drained
        assert not engines[rank].gpu_connector.prepared

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    assert not page_runtime.store.gets and not page_runtime.store.puts


def test_callback_aliases_require_the_same_object(page_runtime: Any) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    callbacks = _callbacks([_request()])
    _bind(engine, backend, [_request()], callbacks + (callbacks[0], callbacks[-1]))
    backend.wait_for_load(callbacks[0])
    assert backend._callbacks[0, 0] is callbacks[0]
    backend.abort_step()


@pytest.mark.parametrize("phase", ["wait", "submit", "finish"])
def test_previous_step_same_generation_callback_never_consumes_current_row(
    page_runtime: Any, phase: str
) -> None:
    engines, backends = _tp_backends(page_runtime)
    old = [_callbacks([_request()]) for _ in engines]
    page_runtime.parallel(
        *[
            lambda rank=rank: _execute(
                engines[rank], backends[rank], [_request()], callbacks=old[rank]
            )
            for rank in (0, 1)
        ]
    )
    req = _request(start=300, end=530)  # Same allocation generation, different step.

    def run(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        current = _bind(engine, backend, [req])[0]
        assert current == old[rank][0] and current is not old[rank][0]
        assert backend._previous_callbacks[0] is old[rank][0]
        if phase == "wait":
            with pytest.raises(ValueError, match="registered|identity mismatch"):
                backend.wait_for_load(old[rank][0] if rank == 1 else current)
        else:
            backend.wait_for_load(current)
            record = backend._rows[0, 0]
            _compute(engine, [req], current)
            backend.submit_save(
                old[rank][0] if rank == 1 and phase == "submit" else current,
                _registry(engine)[current.row.layer_name],
            )
            with pytest.raises(ValueError, match="registered|identity mismatch"):
                backend.finish_save(
                    old[rank][0] if rank == 1 and phase == "finish" else current
                )
            assert backend._rows[0, 0] is record and not record.finished
            assert all(obj.is_valid() for obj in record.sources[0].fresh)
        assert backend._saved == [0, 0]
        assert backend._abort_drained
        backend.abort_step()  # No repeated ACK after the common failure handshake.

    page_runtime.parallel(lambda: run(0), lambda: run(1))


def test_reused_callback_object_rejected_at_next_bind_all_tp(page_runtime: Any) -> None:
    engines, backends = _tp_backends(page_runtime)
    callbacks = [_callbacks([_request()]) for _ in engines]
    page_runtime.parallel(
        *[
            lambda rank=rank: _execute(
                engines[rank], backends[rank], [_request()], callbacks=callbacks[rank]
            )
            for rank in (0, 1)
        ]
    )

    def bind(rank: int) -> None:
        req = _request(start=300, end=530)
        before = len(engines[rank].gpu_connector.prepared)
        with pytest.raises(ValueError, match="reused"):
            _bind(
                engines[rank],
                backends[rank],
                [req],
                callbacks[rank] if rank == 1 else _callbacks([req]),
            )
        assert len(engines[rank].gpu_connector.prepared) == before

    page_runtime.parallel(lambda: bind(0), lambda: bind(1))


@pytest.mark.parametrize("unsafe", [False, True])
def test_abort_meets_finish_ack_drains_lookahead_once_before_any_release(
    page_runtime: Any, monkeypatch: Any, unsafe: bool
) -> None:
    engines, backends = _tp_backends(page_runtime)
    page_runtime.parallel(
        *[
            lambda rank=rank: _execute(engines[rank], backends[rank], [_request()])
            for rank in (0, 1)
        ]
    )
    req = _request(start=300, end=530)
    callbacks = [None, None]

    def prepare(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        callbacks[rank] = _bind(engine, backend, [req])[:2]
        for metadata in callbacks[rank]:
            backend.wait_for_load(metadata)
            _compute(engine, [req], metadata)
            backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
        backend.submit_load(callbacks[rank][0])

    page_runtime.parallel(lambda: prepare(0), lambda: prepare(1))
    assert all(
        any(
            t.submitted and not t.complete and not t.kwargs["direction"]
            for t in engine.gpu_connector.tickets
        )
        for engine in engines
    )
    acknowledgements = [[], []]
    for rank, engine in enumerate(engines):
        acknowledge = engine.layerwise_prefill_ack

        def ack(
            identity: Any,
            error: Any = None,
            *,
            rank: int = rank,
            acknowledge: Any = acknowledge,
        ) -> None:
            acknowledgements[rank].append(identity)
            acknowledge(identity, error)

        monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
    entered, gate = Event(), Event()
    drain = engines[1].gpu_connector.drain_layerwise_prefill_transfers

    def delayed_drain() -> None:
        entered.set()
        assert gate.wait(10)
        engines[1].gpu_connector.fence_error = unsafe
        drain()

    monkeypatch.setattr(
        engines[1].gpu_connector, "drain_layerwise_prefill_transfers", delayed_drain
    )
    released = []
    for backend in backends:
        release = backend._release

        def release_after_drain(
            objects: list, *, release: Any = release, backend: Any = backend
        ) -> None:
            assert not unsafe
            # Peers may not yet have returned from the completed collective to
            # update their Python flags; their actual device work is all fenced.
            assert backend._abort_drained
            assert all(
                t.complete
                for e in engines
                for t in e.gpu_connector.tickets
                if t.submitted
            )
            released.extend(objects)
            release(objects)

        monkeypatch.setattr(backend, "_release", release_after_drain)

    def abort() -> None:
        with pytest.raises(
            LayerwisePrefillFenceError if unsafe else ValueError,
            match="Cannot release" if unsafe else "identity mismatch",
        ):
            backends[0].abort_step()

    def finish() -> None:
        with pytest.raises(
            LayerwisePrefillFenceError if unsafe else ValueError,
            match="unknown" if unsafe else "identity mismatch",
        ):
            backends[1].finish_save(callbacks[1][0])
        if not unsafe:
            backends[1].abort_step()

    with ThreadPoolExecutor(max_workers=1) as pool:
        completed = pool.submit(page_runtime.parallel, abort, finish)
        assert entered.wait(5)
        assert not completed.done() and not released
        assert all(not b._step_future.done() for b in backends)
        gate.set()
        completed.result(15)
    assert all(
        len(acks) == 2 and acks[1] == ("failed_step_drained", 2)
        for acks in acknowledgements
    )
    if unsafe:
        for backend in backends:
            with pytest.raises(LayerwisePrefillFenceError, match="Cannot release"):
                backend.abort_step()
        assert not released and all(
            b._unsafe_transfer and not b._abort_drained for b in backends
        )
        monkeypatch.setattr(
            engines[1].gpu_connector, "drain_layerwise_prefill_transfers", drain
        )
        for backend in backends:
            monkeypatch.setattr(
                backend, "_release", LayerwisePrefillAsyncBackend._release
            )
        _dispose_cpu_quarantine(engines, backends)
    else:
        assert released and all(not b._bound and not b._history for b in backends)
    assert all(len(acks) == 2 for acks in acknowledgements)


def test_broken_initial_ack_never_attempts_failure_or_abort_collective(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engines, backends = _tp_backends(page_runtime)
    callbacks = [None, None]

    def prepare(rank: int) -> None:
        callbacks[rank] = _bind(engines[rank], backends[rank], [_request()])[:2]
        for metadata in callbacks[rank]:
            backends[rank].wait_for_load(metadata)
            backends[rank].submit_save(
                metadata, _registry(engines[rank])[metadata.row.layer_name]
            )
        backends[rank].submit_load(callbacks[rank][0])

    page_runtime.parallel(lambda: prepare(0), lambda: prepare(1))
    acknowledgements = []
    for engine in engines:

        def broken(identity: Any, error: Any = None) -> None:
            acknowledgements.append(identity)
            raise LayerwisePrefillFenceError("broken collective sentinel")

        monkeypatch.setattr(engine, "layerwise_prefill_ack", broken)

    def run(rank: int) -> None:
        backend = backends[rank]
        with pytest.raises(LayerwisePrefillFenceError, match="broken collective"):
            backend.finish_save(callbacks[rank][0])
        for _ in range(2):
            with pytest.raises(LayerwisePrefillFenceError, match="Cannot release"):
                backend.abort_step()
        assert backend._rows and not backend._abort_drained

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    assert len(acknowledgements) == 2
    _dispose_cpu_quarantine(engines, backends)
    assert len(acknowledgements) == 2


@pytest.mark.parametrize("cleanup", ["completed", "stale_participant", "unknown"])
def test_retiring_other_request_never_touches_bound_work(
    page_runtime: Any, cleanup: str
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _execute(engine, backend, [_request(request_id="seed")])
    req = _request(1, generation=2, request_id="external")
    metadata = _bind(engine, backend, [req])[0]
    backend.wait_for_load(metadata)
    _compute(engine, [req], metadata)
    backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
    row, future = backend._rows[0, 0], backend._step_future
    events = list(engine.gpu_connector.events)
    if cleanup == "completed":
        backend.abort_request("seed", allocation_generation=1)
        assert "seed" not in backend._history
    elif cleanup == "stale_participant":
        backend.abort_request("external", allocation_generation=1)
    else:
        backend.abort_request("missing")
    assert engine.gpu_connector.events == events
    assert backend._rows[0, 0] is row and backend._step_future is future
    assert not future.done() and not backend._poisoned
    assert backend.pending_jobs() == 1 and backend._bound
    assert backend.finish_save(metadata) is future
    backend.abort_step()


def test_cancelling_participant_retires_whole_batch_not_completed_history(
    page_runtime: Any,
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _execute(engine, backend, [_request(2)])
    requests = [_request(0), _request(1)]
    metadata = _bind(engine, backend, requests)[0]
    backend.wait_for_load(metadata)
    backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
    backend.abort_request("req-0", allocation_generation=1)
    assert not backend._rows and not backend._requests and not backend._bound
    assert set(backend._history) == {"req-2"}
    assert {identity[0] for identity in backend._prefixes} == {"req-2"}
    assert backend.pending_jobs() == 0 and backend._step_future.exception() is not None
    backend.abort_request("req-1", allocation_generation=1)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["admission", "close", "local_put"])
def test_storage_baseexception_is_normalized_drained_and_acknowledged_all_tp(
    page_runtime: Any, monkeypatch: Any, interruption: type, stage: str
) -> None:
    engines, backends = _tp_backends(page_runtime)
    for backend in backends:
        backend.configure_window_limits(8, 64 << 20, 1 if stage == "admission" else 8)
    requests = [_request()]
    callbacks = [None, None]
    if stage == "local_put":

        def prepare(rank: int) -> None:
            callbacks[rank] = _bind(engines[rank], backends[rank], requests)[0]
            backends[rank].wait_for_load(callbacks[rank])
            backends[rank].submit_save(
                callbacks[rank],
                _registry(engines[rank])[callbacks[rank].row.layer_name],
            )

        page_runtime.parallel(lambda: prepare(0), lambda: prepare(1))
    else:
        page_runtime.parallel(
            *[
                lambda rank=rank: _execute(
                    engines[rank], backends[rank], requests, finish=False
                )
                for rank in (0, 1)
            ]
        )
    manager = engines[0].storage_manager
    put = manager.batched_put_sync_required
    remote_ready = Barrier(2, timeout=10)
    interrupted_calls = []
    for name in ("submit", "close"):
        method = getattr(RequiredPutQueue, name)

        def queue_call(
            self: Any, *args: Any, name: str = name, method: Any = method, **kwargs: Any
        ) -> Any:
            try:
                return method(self, *args, **kwargs)
            except BaseException as exc:
                interrupted_calls.append((name, type(exc)))
                raise

        monkeypatch.setattr(RequiredPutQueue, name, queue_call)

    def interrupted_put(keys: Any, objects: Any, **kwargs: Any) -> None:
        put(keys, objects, **kwargs)  # Real strict put consumes every borrowed ref.
        if kwargs.get("location") == (
            "LocalCPUBackend" if stage == "local_put" else "RemoteBackend"
        ):
            if stage == "close":
                # Both jobs must be admitted before either worker reports its
                # interruption, so this specifically exercises close(), not a
                # racing failure of the second submit().
                remote_ready.wait()
            raise interruption("storage interruption sentinel")

    monkeypatch.setattr(manager, "batched_put_sync_required", interrupted_put)

    def run(rank: int) -> None:
        backend = backends[rank]
        with pytest.raises(ValueError, match=interruption.__name__):
            if stage == "local_put":
                backend.finish_save(callbacks[rank])
            else:
                backend.finish_step()
        assert backend._abort_drained and not backend._unsafe_transfer
        assert backend.pending_futures() == 0
        assert interruption.__name__ in str(backend._step_future.exception())
        backend.abort_step()

    page_runtime.parallel(lambda: run(0), lambda: run(1))
    if stage != "local_put":
        assert (
            "submit" if stage == "admission" else "close",
            interruption,
        ) in interrupted_calls
    assert all(
        obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
        for obj in manager.local_cpu_backend.hot_cache.values()
    )


def test_warm_load_uses_retained_lists_and_constant_size_ack(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _execute(engine, backend, [_request()])
    plan = backend._plan
    acknowledge = engine.layerwise_prefill_ack
    summaries = []

    def save_plan(req: Any, end: int, group: int, row: int) -> Any:
        assert end == req.compute_end, "Warm load rebuilt the retained key plan"
        return plan(req, end, group, row)

    def compact_ack(identity: Any, error: Any = None) -> None:
        if isinstance(identity[1], tuple) and identity[1][0] == "prepare_load":
            summary = identity[1][-1]
            assert summary == (True, 1, 300, 2, 11)
            assert len(pickle.dumps(identity)) < 160
            summaries.append(summary)
        acknowledge(identity, error)

    monkeypatch.setattr(backend, "_plan", save_plan)
    monkeypatch.setattr(engine, "layerwise_prefill_ack", compact_ack)
    _execute(engine, backend, [_request(start=300, end=530)])
    assert len(summaries) == 101
