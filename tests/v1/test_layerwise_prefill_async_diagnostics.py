# SPDX-License-Identifier: Apache-2.0
"""Process-local GC diagnostics and CPU preparation timing, without NPU work."""

# ruff: noqa: F811

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import gc
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.layerwise_prefill_async import LayerwisePrefillAsyncBackend
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillFenceError,
    LayerwisePrefillSyncBackend,
)
import lmcache_ascend.v1.layerwise_prefill_async as async_module
import lmcache_ascend.v1.layerwise_prefill_sync as sync_module

# Local
from tests.v1.test_layerwise_prefill_async import (
    _backend,
    _bind,
    _dispose_cpu_quarantine,
    _execute,
    _join_publications,
    _protocol_of,
    _tp_backends,
)
from tests.v1.test_layerwise_prefill_sync import (
    _registry,
    _request,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401


@pytest.fixture
def diagnostics(monkeypatch: Any) -> Any:
    clock = [0.0]
    registry = SimpleNamespace(
        callbacks=[],
        isenabled=lambda: True,
        get_threshold=lambda: (700, 10, 10),
    )
    for name in ("collect", "enable", "disable", "freeze", "set_threshold"):
        setattr(registry, name, Mock(side_effect=AssertionError("GC policy changed")))
    monkeypatch.setattr(async_module, "gc", registry, raising=False)
    monkeypatch.setattr(async_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(sync_module, "perf_counter", lambda: clock[0])
    logger = Mock()
    monkeypatch.setattr(async_module, "logger", logger, raising=False)
    yield SimpleNamespace(gc=registry, clock=clock, logger=logger)
    for name in ("collect", "enable", "disable", "freeze", "set_threshold"):
        getattr(registry, name).assert_not_called()


def _collection(callbacks: list, clock: list, generation: int, seconds: float) -> None:
    info = {"generation": generation, "collected": 0, "uncollectable": 0}
    for callback in tuple(callbacks):
        callback("start", info)
    clock[0] += seconds
    for callback in tuple(callbacks):
        callback("stop", info)


@pytest.mark.parametrize("direction", [False, True], ids=["load", "save"])
@pytest.mark.parametrize(
    "site,error_type",
    [(None, None)]
    + [
        (site, error)
        for site in ("view", "connector")
        for error in (ValueError, KeyboardInterrupt, SystemExit)
    ],
)
def test_prepare_counts_exact_tensor_property_and_connector_time_even_on_failure(
    diagnostics: Any, direction: bool, site: str | None, error_type: Any
) -> None:
    backend = object.__new__(LayerwisePrefillAsyncBackend)
    backend._host_timings = {}
    key = ("layer", 0, 0, 0, 1)
    req = SimpleNamespace(request_id="req")
    planes, slots, result = object(), object(), object()
    backend._caches = {"layer": planes}
    backend._slots = {("req", 1, 0): slots}
    error = error_type("prepare sentinel") if error_type else None
    views = []

    class DelayedView:
        @property
        def tensor(self) -> Any:
            diagnostics.clock[0] += 0.125
            if site == "view" and views:
                raise error
            tensor = object()
            views.append(tensor)
            return tensor

    def prepare(*args: Any, **kwargs: Any) -> Any:
        diagnostics.clock[0] += 0.25
        assert args == (planes, views, [0, 1], [1, 2], slots)
        assert kwargs == {"kv_group": 0, "direction": direction}
        if site == "connector":
            raise error
        return result

    connector = Mock(side_effect=prepare)
    backend._engine = SimpleNamespace(
        gpu_connector=SimpleNamespace(prepare_layerwise_prefill_row=connector)
    )

    def attempt() -> Any:
        args = (req, key, [DelayedView(), DelayedView()], [0, 1], [1, 2], direction)
        if direction:
            return backend._timed_call("prepare_save", backend._prepare_ticket, *args)
        return backend._prepare_ticket(*args)

    for _ in range(2):
        views.clear()
        if error is None:
            assert attempt() is result
        else:
            with pytest.raises(error_type) as caught:
                attempt()
            assert caught.value is error
    seconds = 0.25 if site == "view" else 0.5
    assert backend._host_timings == {
        "prepare_save" if direction else "prepare_load": (2, 2 * seconds, seconds)
    }
    assert connector.call_count == (0 if site == "view" else 2)


def test_window_bind_precedes_base_begin_and_elapsed_includes_validation_and_ack(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    validate, ack = backend._callback_key, engine.layerwise_prefill_ack
    seen = False

    def validate_callback(*args: Any) -> Any:
        nonlocal seen
        assert len(diagnostics.gc.callbacks) == 1
        if not seen:
            _collection(diagnostics.gc.callbacks, diagnostics.clock, 0, 0.125)
            seen = True
        return validate(*args)

    def acknowledge(
        identity: tuple, error: Any = None, *, flush: bool = True
    ) -> None:
        if isinstance(identity[1], tuple) and identity[1][0] == "window_bind":
            _collection(diagnostics.gc.callbacks, diagnostics.clock, 2, 0.25)
        elif identity[0] == "bind":
            diagnostics.clock[0] += 0.5
        ack(identity, error, flush=flush)

    monkeypatch.setattr(backend, "_callback_key", validate_callback)
    monkeypatch.setattr(engine, "layerwise_prefill_ack", acknowledge)
    logger = Mock()

    def log_begin(*args: Any) -> None:
        assert diagnostics.clock[0] == 0.375

    logger.info.side_effect = log_begin
    monkeypatch.setattr(sync_module, "logger", logger)
    _bind(engine, backend, [_request()])
    assert backend.window_stats()["async_host"]["window_bind"] == (1, 375.0, 375.0)
    assert backend._step_started == 0.375
    assert backend._bind_seconds == 0.5
    assert backend.window_stats()["gc"]["counts"] == (1, 0, 1)
    assert "event=begin" in logger.info.call_args.args[0]
    backend.abort_step()
    assert not diagnostics.gc.callbacks


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "site", ["validation", "window_ack", "base", "base_ack", "reset"]
)
def test_bind_failure_detaches_even_raw_baseexception(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any, site: str, error_type: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    error = error_type("bind sentinel")

    def fail(*args: Any, **kwargs: Any) -> None:
        assert len(diagnostics.gc.callbacks) == 1
        raise error

    if site == "validation":
        monkeypatch.setattr(backend, "_callback_key", fail)
    elif site == "base":
        monkeypatch.setattr(LayerwisePrefillSyncBackend, "bind_step", fail)
    elif site == "reset":
        monkeypatch.setattr(engine, "reset_layerwise_prefill_ack_stats", fail)
    else:
        ack = engine.layerwise_prefill_ack

        def acknowledge(
            identity: tuple, error: Any = None, *, flush: bool = True
        ) -> None:
            phase = identity[1][0] if isinstance(identity[1], tuple) else identity[0]
            if phase == ("window_bind" if site == "window_ack" else "bind"):
                fail()
            ack(identity, error, flush=flush)

        monkeypatch.setattr(engine, "layerwise_prefill_ack", acknowledge)
    with pytest.raises(error_type, match="bind sentinel") as caught:
        _bind(engine, backend, [_request()])
    if error_type is not ValueError:
        assert caught.value is error
    assert not diagnostics.gc.callbacks
    assert not diagnostics.logger.method_calls
    assert not engine.gpu_connector.tickets


def test_per_generation_stats_are_bounded_reset_and_snapshots_are_independent(
    page_runtime: Any, diagnostics: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    sentinel = Mock()
    diagnostics.gc.callbacks.append(sentinel)
    for step in (1, 2):
        _bind(engine, backend, [])
        assert len(diagnostics.gc.callbacks) == 2
        callback = diagnostics.gc.callbacks[-1]
        callback("stop", {"generation": 2})  # No start in this step.
        for generation in range(3):
            for seconds in (0.125 * (generation + 1), 0.25 * (generation + 1)):
                _collection(
                    diagnostics.gc.callbacks, diagnostics.clock, generation, seconds
                )
        expected = {
            "enabled": True,
            "threshold": (700, 10, 10),
            "counts": (2, 2, 2),
            "total_ms": (375.0, 750.0, 1125.0),
            "max_ms": (250.0, 500.0, 750.0),
        }
        stats = backend.window_stats()["gc"]
        assert stats == expected
        stats["counts"] = ()
        assert backend.window_stats()["gc"] == expected
        backend.finish_step()
        assert diagnostics.gc.callbacks == [sentinel]
        assert backend._step == step
        assert backend.window_stats()["gc"] == expected
    assert sentinel.call_count == 24
    assert not diagnostics.logger.method_calls  # Root uses only the existing record.


def test_rejected_rebind_never_duplicates_or_restarts_diagnostics(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    callback = diagnostics.gc.callbacks[0]
    _collection([callback], diagnostics.clock, 1, 0.125)
    acknowledge = engine.layerwise_prefill_ack

    def ack(*args: Any, **kwargs: Any) -> None:
        assert diagnostics.gc.callbacks == [callback]
        acknowledge(*args, **kwargs)

    monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
    with pytest.raises(ValueError, match="unfinished"):
        _bind(engine, backend, [])
    assert not diagnostics.gc.callbacks
    assert backend.window_stats()["gc"]["counts"] == (0, 1, 0)
    backend.abort_step()
    with pytest.raises(ValueError, match="poisoned"):
        _bind(engine, backend, [])
    assert not diagnostics.gc.callbacks


def test_callback_removal_uses_identity_not_third_party_equality(
    page_runtime: Any, diagnostics: Any
) -> None:
    class EqualCallback:
        def __eq__(self, other: Any) -> bool:
            return True

        def __call__(self, phase: str, info: dict) -> None:
            pass

    engine = page_runtime.engine()
    backend = _backend(engine)
    sentinel = EqualCallback()
    diagnostics.gc.callbacks.append(sentinel)
    _bind(engine, backend, [])
    assert len(diagnostics.gc.callbacks) == 2
    backend.finish_step()
    assert len(diagnostics.gc.callbacks) == 1
    assert diagnostics.gc.callbacks[0] is sentinel


@pytest.mark.parametrize(
    "terminal", ["abort", "fail_ack", "broken_ack", "finish_error"]
)
def test_terminal_error_cleanup_is_idempotent_and_preserves_failure_ownership(
    page_runtime: Any, diagnostics: Any, terminal: str
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    metadata = _bind(engine, backend, [_request()])[0]
    backend.wait_for_load(metadata)
    owner = backend._rows[0, 0].sources[0].fresh[0]
    if terminal == "abort":
        backend.abort_step()
        assert not backend._rows and not backend._bound
    else:
        with pytest.raises(
            LayerwisePrefillFenceError if terminal == "broken_ack" else ValueError
        ):
            if terminal == "finish_error":
                backend.finish_step()
            else:
                backend._fail_ack(
                    RuntimeError("broken")
                    if terminal == "broken_ack"
                    else ValueError("failed")
                )
        assert owner.is_valid() and owner.metadata.pin_count == 1
        assert backend._step_future.done() and backend._step_future.exception()
    assert not diagnostics.gc.callbacks
    if terminal == "broken_ack":
        with pytest.raises(LayerwisePrefillFenceError):
            backend.abort_step()
        _dispose_cpu_quarantine([engine], [backend])
    else:
        backend.abort_step()
    assert not diagnostics.gc.callbacks


@pytest.mark.parametrize(
    "terminal", ["finish_save", "finish_step", "abort_step", "already_unsafe"]
)
def test_unknown_fence_detaches_without_releasing_any_owner(
    page_runtime: Any, diagnostics: Any, terminal: str
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    metadata = _bind(engine, backend, [_request()])[0]
    backend.wait_for_load(metadata)
    backend.submit_save(metadata, _registry(engine)[metadata.row.layer_name])
    owner = backend._rows[0, 0].sources[0].fresh[0]
    engine.gpu_connector.fence_error = True
    if terminal == "already_unsafe":
        backend._unsafe_transfer = True
        terminal = "abort_step"
    try:
        if terminal != "finish_save" or _protocol_of(backend) is None:
            with pytest.raises(LayerwisePrefillFenceError):
                getattr(backend, terminal)(
                    *([metadata] if terminal == "finish_save" else [])
                )
        else:
            # Plan B defers the row raise to the next entry; the unknown fence
            # is already recorded locally and the owners stay quarantined.
            backend.finish_save(metadata)
            _join_publications(backend)
            assert isinstance(backend._pending_error, LayerwisePrefillFenceError)
        assert not diagnostics.gc.callbacks
        assert backend._unsafe_transfer and backend._rows and backend._prefixes
        assert owner.is_valid() and owner.metadata.pin_count == 1
        with pytest.raises(LayerwisePrefillFenceError):
            backend.abort_step()
        assert not diagnostics.gc.callbacks
    finally:
        _dispose_cpu_quarantine([engine], [backend])


@pytest.mark.parametrize("terminal", ["finish_step", "abort_step", "_fail_ack"])
def test_raw_terminal_interrupt_still_detaches(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any, terminal: str
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    error = KeyboardInterrupt("drain sentinel")
    with monkeypatch.context() as patch:
        patch.setattr(backend, "_drain_devices", Mock(side_effect=error))
        with pytest.raises(KeyboardInterrupt) as caught:
            getattr(backend, terminal)(
                *([ValueError("failed")] if terminal == "_fail_ack" else [])
            )
        assert caught.value is error
    assert not diagnostics.gc.callbacks
    backend.abort_step()


@pytest.mark.parametrize("seconds", [0.099, 0.1, 0.125])
@pytest.mark.parametrize("terminal", ["finish_step", "abort_step", "_fail_ack"])
@pytest.mark.parametrize("rank", [0, 3, None])
def test_passive_record_only_once_at_threshold_on_terminal(
    page_runtime: Any,
    diagnostics: Any,
    monkeypatch: Any,
    seconds: float,
    terminal: str,
    rank: int | None,
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    # Empty steps need no passive slab. Metadata alone controls diagnostic rank.
    monkeypatch.setattr(
        engine,
        "metadata",
        SimpleNamespace(
            is_first_rank=lambda: rank == 0,
            world_size=1,
            **({"worker_id": rank} if rank is not None else {}),
        ),
    )
    _collection(diagnostics.gc.callbacks, diagnostics.clock, 2, seconds)
    assert not diagnostics.logger.method_calls
    if terminal == "_fail_ack":
        with pytest.raises(ValueError):
            backend._fail_ack(ValueError("failed"))
    else:
        getattr(backend, terminal)()
    assert not diagnostics.gc.callbacks
    backend.abort_step()
    expected = int(rank != 0 and seconds >= 0.1)
    assert diagnostics.logger.info.call_count == expected
    if expected:
        message, *args = diagnostics.logger.info.call_args.args
        line = message % tuple(args)
        assert "step=1" in line and f"rank={rank}" in line
        assert (
            ast.literal_eval(line.split(" gc=", 1)[1]) == backend.window_stats()["gc"]
        )


def test_diagnostic_failures_do_not_replace_transfer_errors_or_prevent_cleanup(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    callback = diagnostics.gc.callbacks[0]
    with monkeypatch.context() as patch:
        patch.setattr(async_module, "perf_counter", Mock(side_effect=KeyboardInterrupt))
        callback("start", {"generation": 0})
        callback("stop", {"generation": 0})
    for info in ({}, {"generation": -1}, {"generation": 3}, {"generation": "0"}):
        callback("start", info)
        callback("stop", info)
    _collection(diagnostics.gc.callbacks, diagnostics.clock, 2, 0.125)
    monkeypatch.setattr(engine.metadata, "is_first_rank", lambda: False)
    diagnostics.logger.info.side_effect = KeyboardInterrupt("diagnostic logger")
    failure = ValueError("transfer sentinel")
    with pytest.raises(ValueError) as caught:
        backend._fail_ack(failure)
    assert caught.value is failure
    assert not diagnostics.gc.callbacks
    backend.abort_step()


def test_gc_snapshot_and_passive_logger_failure_cannot_fail_successful_finish(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    _collection(diagnostics.gc.callbacks, diagnostics.clock, 2, 0.125)
    backend._gc.total[0] = object()  # Simulate a diagnostic formatting failure.
    assert backend.window_stats()["gc"] == {}
    monkeypatch.setattr(engine.metadata, "is_first_rank", lambda: False)
    diagnostics.logger.info.side_effect = KeyboardInterrupt("diagnostic logger")
    backend.finish_step()
    assert backend._step_future.result() is None and not backend._bound
    assert not diagnostics.gc.callbacks


def test_registration_failure_is_best_effort(
    page_runtime: Any, diagnostics: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    _bind(engine, backend, [])
    _collection(diagnostics.gc.callbacks, diagnostics.clock, 2, 0.125)
    backend.finish_step()
    diagnostics.gc.get_threshold = Mock(
        side_effect=KeyboardInterrupt("diagnostic setup")
    )
    _bind(engine, backend, [])
    backend.finish_step()
    assert not diagnostics.gc.callbacks
    assert backend.window_stats()["gc"] == {}  # Never report the preceding step.


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_discarded_backend_does_not_leak_global_callback_or_retain_backend(
    page_runtime: Any, diagnostics: Any, partial: bool, cyclic: bool
) -> None:
    engine = page_runtime.engine()
    for _ in range(3):
        backend = _backend(engine)
        _bind(engine, backend, [])
        callback = diagnostics.gc.callbacks[0]
        if partial:
            callback("start", {"generation": 2})
        if cyclic:
            backend.cycle = backend
        reference = weakref.ref(backend)
        engine._layerwise_prefill_window_backend = None
        del backend
        if cyclic:
            gc.collect()  # Only the test drives collection of the discarded cycle.
        assert reference() is None
        assert not diagnostics.gc.callbacks
        # A snapshot of the callback registry may outlive its backend.
        callback("stop", {"generation": 2})
    assert not diagnostics.logger.method_calls


def test_disabled_gc_manual_collections_preserve_global_policy_and_callbacks(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any
) -> None:
    engine = page_runtime.engine()
    backend = _backend(engine)
    original_callbacks = tuple(gc.callbacks)
    enabled, threshold = gc.isenabled(), gc.get_threshold()
    seen = []

    def sentinel(phase: str, info: dict) -> None:
        seen.append((phase, info["generation"]))
        if phase == "start":
            diagnostics.clock[0] += 0.125

    gc.disable()  # Test policy only; production must not enable it.
    try:
        monkeypatch.setattr(async_module, "gc", gc)
        _bind(engine, backend, [])
        assert len(gc.callbacks) == len(original_callbacks) + 1
        # A later callback advances the fake clock inside each collection.
        gc.callbacks.append(sentinel)
        for generation in range(3):
            gc.collect(generation)
        stats = backend.window_stats()["gc"]
        assert stats == {
            "enabled": False,
            "threshold": threshold,
            "counts": (1, 1, 1),
            "total_ms": (125.0, 125.0, 125.0),
            "max_ms": (125.0, 125.0, 125.0),
        }
        backend.finish_step()
        assert tuple(gc.callbacks) == (*original_callbacks, sentinel)
        assert not gc.isenabled() and gc.get_threshold() == threshold
        assert seen == [
            (phase, generation)
            for generation in range(3)
            for phase in ("start", "stop")
        ]
    finally:
        if backend._bound:
            backend.abort_step()
        if sentinel in gc.callbacks:
            gc.callbacks.remove(sentinel)
        if enabled:
            gc.enable()
    assert tuple(gc.callbacks) == original_callbacks


@pytest.mark.parametrize("slow_peer", [False, True])
def test_two_tp_steps_keep_five_gate_acks_root_log_count_and_no_extra_device_work(
    page_runtime: Any, diagnostics: Any, monkeypatch: Any, slow_peer: bool
) -> None:
    engines, backends = _tp_backends(page_runtime)
    logger = Mock()
    monkeypatch.setattr(sync_module, "logger", logger)
    gather = Mock(wraps=torch.distributed.all_gather)
    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    object_gather = Mock(wraps=torch.distributed.all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_gather_object", object_gather)
    stream_syncs = []
    for engine in engines:
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
    slow_calls = 0
    for step, (start, end) in enumerate(((0, 300), (300, 530)), 1):
        requests = [_request(start=start, end=end, generation=step)]
        futures = page_runtime.parallel(
            *[
                lambda rank=rank, requests=requests: _execute(
                    engines[rank], backends[rank], requests, finish=False
                )
                for rank in (0, 1)
            ]
        )
        assert len(diagnostics.gc.callbacks) == 2
        if slow_peer:
            # The threaded TP fixture shares a process; inject into only the
            # passive callback to model a GC pause in a separate rank process.
            _collection([backends[1]._gc], diagnostics.clock, 2, 0.125)
        page_runtime.parallel(*(backend.finish_step for backend in backends))
        assert all(future.result() is None for future in futures)
        assert not diagnostics.gc.callbacks
        assert diagnostics.logger.info.call_count == (step if slow_peer else 0)
        assert logger.info.call_count == 2 * step
        # Plan A: five per-step gate collectives per rank; rows issue none.
        assert gather.call_count == 2 * 5 * step
        slow_calls += sum(
            engine.layerwise_prefill_ack_stats()["slow_count"] for engine in engines
        )
        assert object_gather.call_count == slow_calls
        assert all(sync.call_count == step for sync in stream_syncs)
        for rank, backend in enumerate(backends):
            assert backend.window_stats()["ack"]["calls"] == 5
            assert backend.window_stats()["gc"]["counts"] == (
                0,
                0,
                int(rank == 1 and slow_peer),
            )
            assert backend.window_stats()["async_host"]["window_bind"] == (1, 0.0, 0.0)
        message, *args = logger.info.call_args.args
        line = message % tuple(args)
        assert (
            ast.literal_eval(line.split(" window_stats=", 1)[1])
            == backends[0].window_stats()
        )
