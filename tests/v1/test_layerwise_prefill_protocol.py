# SPDX-License-Identifier: Apache-2.0
"""Plan B protocol thread: single emitter, deferred publication, capacity."""

# Standard
from threading import Event, current_thread
from typing import Any

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.layerwise_prefill_async import LayerwisePrefillAsyncBackend
from lmcache_ascend.v1.layerwise_prefill_protocol import (
    LayerwisePrefillProtocolThread,
)

# Local
from tests.v1.test_layerwise_prefill_async import (
    _backend,
    _bind,
    _compute,
    _execute,
    _registry,
    _request,
    _tp_backends,
)
from tests.v1.test_layerwise_prefill_sync import runtime  # noqa: F401
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401


@pytest.fixture
def protocol_runtime(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Any:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PROTOCOL_THREAD", "true")
    return request.getfixturevalue("page_runtime")


def test_flag_off_constructs_no_protocol_thread(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PROTOCOL_THREAD", raising=False)
    engine = protocol_runtime.engine()
    backend = _backend(engine)
    assert backend._protocol is None
    assert type(backend) is LayerwisePrefillAsyncBackend


def test_publication_and_gates_leave_the_model_thread(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = protocol_runtime.engine()
    backend = _backend(engine)
    model = current_thread()
    threads = {"resolve": set(), "publish": set()}
    acknowledge = engine.layerwise_prefill_ack
    monkeypatch.setattr(
        engine,
        "layerwise_prefill_ack",
        lambda *args, **kwargs: (
            threads.setdefault("ack", set()).add(current_thread()),
            acknowledge(*args, **kwargs),
        )[1],
    )
    for name in ("resolve_layerwise_prefill_row",):
        resolve = getattr(engine, name)
        monkeypatch.setattr(
            engine,
            name,
            lambda *args, _resolve=resolve, **kwargs: (
                threads["resolve"].add(current_thread()),
                _resolve(*args, **kwargs),
            )[1],
        )
    put = engine.storage_manager.batched_put_sync_required
    monkeypatch.setattr(
        engine.storage_manager,
        "batched_put_sync_required",
        lambda *args, **kwargs: (
            threads["publish"].add(current_thread()),
            put(*args, **kwargs),
        )[1],
    )
    _execute(engine, backend, [_request()])
    stats = backend.window_stats()
    assert stats["protocol"]["cancelled"] is False
    assert stats["ack"]["calls"] == 5
    for phase in ("resolve", "publish"):
        assert threads[phase], phase
        assert model not in threads[phase]
    # window_bind/bind stay on the model thread; the step gates do not.
    assert threads["ack"] == {model, backend._protocol._thread}


def test_bootstrap_resolution_order_is_queue_order(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = protocol_runtime.engine()
    backend = _backend(engine)
    submitted = []
    submit = LayerwisePrefillProtocolThread.submit

    def recording(self: Any, kind: str, payload: Any = None) -> Any:
        submitted.append(kind)
        return submit(self, kind, payload)

    monkeypatch.setattr(LayerwisePrefillProtocolThread, "submit", recording)
    _execute(engine, backend, [_request()])
    # The first execution's four bootstrap resolutions — including both
    # deferred lookaheads — all enter the queue before the first publication.
    assert submitted[:4] == ["resolve"] * 4 and submitted[4] == "publish"
    assert "resolve" in submitted[4:]
    assert submitted.count("finish_step") == 1
    assert not backend._pending_resolves


def test_publication_capacity_bounds_the_admitted_window(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concurrent.futures
    import time
    from dataclasses import replace as _replace

    engine = protocol_runtime.engine()
    backend = _backend(engine)
    backend.configure_window_limits(2, 64 << 20, 2)
    # Bootstrap a full first step so the bounded step runs warm: no deferred
    # row resolution can couple the model thread to the gated queue head.
    _execute(engine, backend, [_request()])
    requests = [
        _replace(
            _request(),
            compute_start=300,
            restore_end=300,
            compute_end=530,
            allocation_generation=2,
        )
    ]
    registry = _registry(engine)
    callbacks = _bind(engine, backend, requests)
    for metadata in callbacks[:2]:
        backend.wait_for_load(metadata)
        _compute(engine, requests, metadata)
        backend.submit_save(metadata, registry[metadata.row.layer_name])
    put = engine.storage_manager.batched_put_sync_required
    release = Event()

    def gated_put(keys: Any, objects: Any, **kwargs: Any) -> None:
        release.wait(10)
        return put(keys, objects, **kwargs)

    monkeypatch.setattr(engine.storage_manager, "batched_put_sync_required", gated_put)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.finish_save, callbacks[0])
        second = pool.submit(backend.finish_save, callbacks[1])
        assert first.result(10) is backend._step_future
        assert second.result(10) is backend._step_future
        # Both execution rows are admitted; the next group's row must block in
        # admission while both gated publications hold the two-job window.
        third = callbacks[2]
        backend.wait_for_load(third)
        _compute(engine, requests, third)
        backend.submit_save(third, registry[third.row.layer_name])
        finishing = pool.submit(backend.finish_save, third)
        time.sleep(0.3)
        assert not finishing.done()
        assert backend._protocol.stats()["publish_jobs"] == 2
        release.set()
        finishing.result(10)
    backend._protocol.join_publications()
    assert backend.pending_jobs() == 0


def test_root_publication_error_rides_envelope_cancels_queue_all_tp(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engines, backends = _tp_backends(protocol_runtime)
    put = engines[0].storage_manager.batched_put_sync_required

    def failing_put(keys: Any, objects: Any, **kwargs: Any) -> None:
        put(keys, objects, **kwargs)
        raise RuntimeError("publication storage died")

    monkeypatch.setattr(
        engines[0].storage_manager, "batched_put_sync_required", failing_put
    )
    requests = [_request()]
    failures = protocol_runtime.parallel(
        *[
            lambda rank=rank: _failing_step(engines[rank], backends[rank], requests)
            for rank in (0, 1)
        ]
    )
    assert failures == [True, True]
    for backend in backends:
        assert backend._protocol.stats()["cancelled"] is True
        assert "publication storage died" in str(backend._pending_error)
        assert backend._abort_drained and not backend._unsafe_transfer


def _failing_step(engine: Any, backend: Any, requests: list) -> bool:
    registry = _registry(engine)
    callbacks = _bind(engine, backend, requests)
    surfaced = False
    for metadata in callbacks[:2]:
        try:
            backend.wait_for_load(metadata)
            _compute(engine, requests, metadata)
            backend.submit_save(metadata, registry[metadata.row.layer_name])
            backend.finish_save(metadata)
        except ValueError:
            surfaced = True
            break
        backend._protocol.join_publications()
        if backend._pending_error is not None:
            surfaced = True
            break
    backend.abort_request(requests[0].request_id)
    return surfaced


def test_gate_failure_completes_drain_handshake_all_tp(
    protocol_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engines, backends = _tp_backends(protocol_runtime)
    requests = [_request()]
    outcomes = []

    def run(rank: int) -> None:
        engine, backend = engines[rank], backends[rank]
        if rank == 1:
            # A rank-local ValueError at the device_finish gate mismatches the
            # batch on every rank; both protocol threads then complete the
            # common failure-drain rendezvous before surfacing.
            monkeypatch.setattr(
                backend,
                "_drain_devices",
                lambda: ValueError("rank-local drain sentinel"),
            )
        with pytest.raises(ValueError):
            _execute(engine, backend, requests)
        assert backend._abort_drained and not backend._unsafe_transfer
        outcomes.append(rank)

    protocol_runtime.parallel(lambda: run(0), lambda: run(1))
    assert sorted(outcomes) == [0, 1]


def test_close_joins_the_emitter(protocol_runtime: Any) -> None:
    engine = protocol_runtime.engine()
    backend = _backend(engine)
    thread = backend._protocol._thread
    assert thread.is_alive()
    _bind(engine, backend, [])
    backend.close()
    assert not thread.is_alive()
    assert backend._protocol is None
