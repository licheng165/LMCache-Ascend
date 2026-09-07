# SPDX-License-Identifier: Apache-2.0
"""Stage 4 NPU transfer-window backend tests."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LayerwisePrefillWindowCoordinator,
    _DSAKVTopologyCache,
)
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.dsa_kv_topology import validate_dsa_kv_topology
from lmcache_ascend.v1.layerwise_prefill_window import (
    LayerwisePrefillDeviceOps,
    LayerwisePrefillNPUWindowBackend,
)
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)

# Local
from tests.v1.test_dsa_kv_topology import _topology


class FakeEvent:
    """Event double tracking record/wait calls."""

    def __init__(self) -> None:
        self.recorded = False
        self.waited = 0

    def record(self, stream: Any = None) -> None:
        self.recorded = True

    def wait(self, stream: Any = None) -> None:
        self.waited += 1


class FakeDeviceOps(LayerwisePrefillDeviceOps):
    """Device-ops double recording every submission."""

    def __init__(
        self,
        *,
        source_banks: int = 2,
        layout_signature: Optional[str] = "glm52-layout",
        future: Any = None,
    ):
        self._source_banks = source_banks
        self._layout_signature = layout_signature
        self._future = future
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def layout_signature(self) -> Optional[str]:
        return self._layout_signature

    def source_bank_count(self) -> int:
        return self._source_banks

    def submit_save(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("submit_save", args, kwargs))

    def submit_load(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("submit_load", args, kwargs))

    def wait_for_load(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("wait_for_load", args, kwargs))

    def finish_publish(self, metadata: Any, done_event: Any) -> Optional[Any]:
        self.calls.append(("finish_publish", (metadata, done_event), {}))
        return self._future

    def sync_save(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("sync_save", args, kwargs))

    def abort_request(self, request_id: str) -> None:
        self.calls.append(("abort_request", (request_id,), {}))


def _row_namespace(key: tuple[Any, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        layer_name=key[0],
        execution_ordinal=key[1],
        kv_group=key[2],
        row_ordinal=key[3],
        bank=key[4],
    )


def _metadata(
    view,
    execution_ordinal: int,
    *,
    row_kind: str = "latent",
    generations: tuple[tuple[str, int], ...] = (("req-1", 7),),
) -> SimpleNamespace:
    execution = view.executions[execution_ordinal]
    latent_key = execution[1]
    indexer_key = execution[2]
    row_key = latent_key if row_kind == "latent" else indexer_key
    assert row_key is not None
    return SimpleNamespace(
        row=_row_namespace(row_key),
        execution=SimpleNamespace(
            execution_ordinal=execution_ordinal,
            latent=_row_namespace(latent_key),
            indexer=(
                None if indexer_key is None else _row_namespace(indexer_key)
            ),
        ),
        request_generations=generations,
    )


def _backend(
    view,
    ops: FakeDeviceOps,
    events: list[FakeEvent],
) -> LayerwisePrefillNPUWindowBackend:
    factory = events.pop
    return LayerwisePrefillNPUWindowBackend(view, ops, event_factory=factory)


def _planes(group: int) -> list[torch.Tensor]:
    count = 2 if group == 0 else 1
    return [torch.empty(4) for _ in range(count)]


@pytest.fixture
def view():
    return validate_dsa_kv_topology(_topology())


def test_backend_reports_topology_cardinality(view) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)

    assert backend.layer_count(0) == 79
    assert backend.layer_count(1) == 22
    assert backend.supports_sync_callbacks is True
    assert backend.supports_transfer_window is True
    assert backend.persists_indexer_group is True
    assert backend.topology_signature == "glm52-topology-signature"


def test_execution_six_lowers_indexer_row_three_bank_one(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(32)]
    backend = _backend(view, ops, events)

    for execution_ordinal in range(7):
        if view.executions[execution_ordinal][2] is None:
            continue
        for kind in ("latent", "indexer"):
            metadata = _metadata(view, execution_ordinal, row_kind=kind)
            group = metadata.row.kv_group
            backend.submit_save(metadata, _planes(group))
            backend.finish_save(metadata)

    submits = [call for call in ops.calls if call[0] == "submit_save"]
    # Execution 6: LATENT row 6 bank 0, INDEXER row 3 bank 1.
    assert submits[-2][1][:3] == (0, 6, 0)
    assert submits[-1][1][:3] == (1, 3, 1)


def test_save_then_finish_publishes_per_bank_event(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)
    metadata = _metadata(view, 0)
    backend.submit_save(metadata, _planes(0))
    backend.finish_save(metadata)

    ledger = backend._ledgers[0][0]
    assert ledger.save_done is not None
    assert ledger.save_done.recorded is False
    publishes = [call for call in ops.calls if call[0] == "finish_publish"]
    assert publishes[0][1][1] is ledger.save_done


def test_load_waits_for_same_bank_previous_save(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)

    backend.submit_save(_metadata(view, 0), _planes(0))
    backend.finish_save(_metadata(view, 0))
    saved_event = backend._ledgers[0][0].save_done

    # Execution 1 targets LATENT row 2 -> bank 0, the bank of row 0.
    backend.submit_load(_metadata(view, 1))

    latent_loads = [
        call
        for call in ops.calls
        if call[0] == "submit_load" and call[1][0] == 0
    ]
    assert len(latent_loads) == 1
    kwargs = latent_loads[0][2]
    assert kwargs["wait_event"] is saved_event
    assert latent_loads[0][1][1] == 2
    assert latent_loads[0][1][2] == 0


def test_group_bank_ledgers_are_isolated(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(16)]
    backend = _backend(view, ops, events)

    backend.submit_save(_metadata(view, 0, row_kind="latent"), _planes(0))
    backend.finish_save(_metadata(view, 0, row_kind="latent"))
    latent_event = backend._ledgers[0][0].save_done

    backend.submit_save(_metadata(view, 0, row_kind="indexer"), _planes(1))
    backend.finish_save(_metadata(view, 0, row_kind="indexer"))
    indexer_event = backend._ledgers[1][0].save_done

    assert latent_event is not indexer_event
    # Each group's load into bank 0 must wait for its own group's save.
    backend.submit_load(_metadata(view, 1))
    loads = [call for call in ops.calls if call[0] == "submit_load"]
    latent_load = next(call for call in loads if call[1][0] == 0)
    indexer_load = next(call for call in loads if call[1][0] == 1)
    assert latent_load[2]["wait_event"] is latent_event
    assert indexer_load[2]["wait_event"] is indexer_event


def test_single_staging_serializes_saves(view) -> None:
    ops = FakeDeviceOps(source_banks=1)
    events = [FakeEvent() for _ in range(16)]
    backend = _backend(view, ops, events)

    backend.submit_save(_metadata(view, 0), _planes(0))
    first_submit = next(
        call for call in ops.calls if call[0] == "submit_save"
    )
    assert first_submit[2]["wait_event"] is None
    backend.finish_save(_metadata(view, 0))

    backend.submit_save(_metadata(view, 1), _planes(0))
    second_submit = [
        call for call in ops.calls if call[0] == "submit_save"
    ][-1]
    # The one staging tensor cannot be reused until the previous D2H event.
    assert second_submit[2]["wait_event"] is first_submit[2]["done_event"]


def test_wait_for_load_includes_bootstrap_and_deduplicates_ready_rows(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)

    backend.wait_for_load(_metadata(view, 0))
    backend.wait_for_load(_metadata(view, 0))
    assert len([call for call in ops.calls if call[0] == "wait_for_load"]) == 1

    backend.submit_load(_metadata(view, 0))
    backend.wait_for_load(_metadata(view, 1))

    waits = [call for call in ops.calls if call[0] == "wait_for_load"]
    assert len(waits) == 2
    assert waits[0][1][:3] == (0, 0, 0)
    assert waits[1][1][:3] == (0, 1, 1)
    # The ledger is consumed; a second wait for the same row no-ops.
    backend.wait_for_load(_metadata(view, 1))
    assert len([call for call in ops.calls if call[0] == "wait_for_load"]) == 2


def test_finish_before_submit_fails_closed(view) -> None:
    backend = _backend(view, FakeDeviceOps(), [FakeEvent()])
    with pytest.raises(ValueError, match="before its submit"):
        backend.finish_save(_metadata(view, 0))


def test_duplicate_submit_fails_closed(view) -> None:
    backend = _backend(view, FakeDeviceOps(), [FakeEvent(), FakeEvent()])
    backend.submit_save(_metadata(view, 0), _planes(0))
    with pytest.raises(ValueError, match="already submitted"):
        backend.submit_save(_metadata(view, 0), _planes(0))


def test_future_generation_finish_does_not_consume_or_activate(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)

    backend.submit_save(_metadata(view, 0, generations=(("req-1", 7),)), _planes(0))
    with pytest.raises(ValueError, match="before its submit"):
        backend.finish_save(_metadata(view, 0, generations=(("req-1", 8),)))

    publishes = [call for call in ops.calls if call[0] == "finish_publish"]
    assert publishes == []
    backend.finish_save(_metadata(view, 0, generations=(("req-1", 7),)))
    assert len([call for call in ops.calls if call[0] == "finish_publish"]) == 1


def test_out_of_range_row_and_wrong_planes_fail_closed(view) -> None:
    backend = _backend(view, FakeDeviceOps(), [FakeEvent(), FakeEvent()])
    bad_row = SimpleNamespace(
        row=SimpleNamespace(
            layer_name="physical.latent.79",
            execution_ordinal=79,
            kv_group=0,
            row_ordinal=79,
            bank=1,
        ),
        execution=SimpleNamespace(),
        request_generations=(("req-1", 7),),
    )
    with pytest.raises(ValueError, match="absent from the frozen NPU layout"):
        backend.submit_save(bad_row, _planes(0))

    with pytest.raises(ValueError, match="wrong KV plane count"):
        backend.submit_save(_metadata(view, 0), _planes(1))


def test_layout_rebind_fails_closed_before_launch(view) -> None:
    ops = FakeDeviceOps()
    backend = _backend(view, ops, [FakeEvent(), FakeEvent()])
    ops._layout_signature = "rebound-layout"
    with pytest.raises(ValueError, match="rebound"):
        backend.submit_save(_metadata(view, 0), _planes(0))


def test_aborted_request_is_refused_before_launch(view) -> None:
    ops = FakeDeviceOps()
    backend = _backend(view, ops, [FakeEvent(), FakeEvent()])
    backend.wait_for_load(_metadata(view, 0))
    backend.abort_request("req-1")
    with pytest.raises(ValueError, match="released"):
        backend.submit_save(_metadata(view, 0), _planes(0))
    assert [call for call in ops.calls if call[0] == "abort_request"]


def test_engine_exposes_backend_only_when_connector_opts_in(view) -> None:
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine._dsa_kv_topology_view = view
    engine._layerwise_prefill_window_backend = None

    class NoOptIn:
        supports_layerwise_prefill_window = lambda self: False

    engine.gpu_connector = NoOptIn()
    assert engine.layerwise_prefill_window_backend is None

    class OptIn:
        def supports_layerwise_prefill_window(self) -> bool:
            return True

    engine.gpu_connector = OptIn()
    assert engine.layerwise_prefill_window_backend is None

    ops = FakeDeviceOps()
    engine.gpu_connector = SimpleNamespace(
        supports_layerwise_prefill_window=lambda: True,
        layerwise_prefill_device_ops=ops,
    )
    backend = engine.layerwise_prefill_window_backend
    assert backend is not None
    assert backend.layer_count(0) == 79
    assert backend.layer_count(1) == 22
    assert engine.layerwise_prefill_window_backend is backend


@pytest.mark.parametrize(
    "hook",
    [
        "sync_save",
        "submit_save",
        "submit_load",
        "wait_for_load",
        "finish_publish",
        "abort_request",
    ],
)
@pytest.mark.parametrize("missing", ["inherited", "noncallable"])
def test_capabilities_require_every_concrete_hook(
    view,
    hook: str,
    missing: str,
) -> None:
    partial_type = type(
        "PartialOps",
        (FakeDeviceOps,),
        {
            hook: getattr(LayerwisePrefillDeviceOps, hook)
            if missing == "inherited"
            else None
        },
    )
    ops = partial_type()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    assert backend.supports_sync_callbacks is (
        hook not in ("sync_save", "wait_for_load", "abort_request")
    )
    assert backend.supports_transfer_window is (hook == "sync_save")

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine._dsa_kv_topology_view = view
    engine.gpu_connector = SimpleNamespace(
        supports_layerwise_prefill_window=lambda: True,
        layerwise_prefill_device_ops=ops,
    )
    assert (engine.layerwise_prefill_window_backend is not None) is (
        hook == "sync_save"
    )
    assert ops.calls == []


def test_base_ops_and_production_connector_do_not_advertise_transport(view) -> None:
    backend = LayerwisePrefillNPUWindowBackend(view, LayerwisePrefillDeviceOps())
    assert backend.supports_sync_callbacks is False
    assert backend.supports_transfer_window is False

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine._dsa_kv_topology_view = view
    engine.gpu_connector = VLLMPagedMemLayerwiseNPUConnector.__new__(
        VLLMPagedMemLayerwiseNPUConnector
    )
    assert engine.layerwise_prefill_window_backend is None


@pytest.mark.parametrize(
    "callback", ["wait_for_load", "sync_save", "submit_save", "submit_load"]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_stale_launch_validation_is_atomic(view, callback: str, mixed: bool) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    backend.wait_for_load(_metadata(view, 0, generations=(("req-1", 9),)))
    generations = (("req-1", 8),)
    if mixed:
        generations = (("fresh", 10), *generations)
    metadata = _metadata(view, 1, generations=generations)
    calls = ops.calls.copy()
    args = (
        (metadata, _planes(0))
        if callback in ("sync_save", "submit_save")
        else (metadata,)
    )

    with pytest.raises(ValueError, match="superseded"):
        getattr(backend, callback)(*args)
    assert ops.calls == calls
    # The first member of an invalid mixed batch must not have been activated.
    backend.wait_for_load(_metadata(view, 0, generations=(("fresh", 7),)))
    backend.wait_for_load(_metadata(view, 0, generations=(("req-1", 9),)))


@pytest.mark.parametrize(
    "generations",
    [(), (("req-1", 10), ("bad", 0)), (("req-1", 10), ("req-1", 11))],
)
def test_malformed_generation_batch_has_no_partial_activation(
    view,
    generations,
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    with pytest.raises(ValueError, match="generations"):
        backend.wait_for_load(_metadata(view, 0, generations=generations))
    assert ops.calls == []
    backend.wait_for_load(_metadata(view, 0))


def test_late_finish_preserves_newer_submission_and_bank_dependencies(view) -> None:
    ops = FakeDeviceOps(source_banks=1)
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    old = _metadata(view, 0)
    new = _metadata(view, 0, generations=(("req-1", 9),))
    backend.submit_save(old, _planes(0))
    backend.finish_save(old)
    backend.submit_save(new, _planes(0))
    current_event = ops.calls[-1][2]["done_event"]
    calls = ops.calls.copy()

    backend.finish_save(old)
    backend.finish_save(_metadata(view, 0, generations=(("req-1", 8),)))
    assert ops.calls == calls
    backend.finish_save(new)
    assert ops.calls[-1][1][1] is current_event
    backend.submit_load(_metadata(view, 1, generations=(("req-1", 9),)))
    latent_load = next(call for call in reversed(ops.calls) if call[1][0] == 0)
    assert latent_load[2]["wait_event"] is current_event


def test_stale_wait_and_matching_finish_leave_current_bank_state_untouched(
    view,
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    old = _metadata(view, 0)
    backend.submit_save(old, _planes(0))
    current = _metadata(view, 0, generations=(("req-1", 9),))
    backend.submit_load(current)
    calls = ops.calls.copy()

    # Even an exact match to the old submit must not publish after supersession.
    backend.finish_save(old)
    with pytest.raises(ValueError, match="superseded"):
        backend.wait_for_load(_metadata(view, 1))
    assert ops.calls == calls
    backend.wait_for_load(_metadata(view, 1, generations=(("req-1", 9),)))
    assert ops.calls[-1][0] == "wait_for_load"
    with pytest.raises(ValueError, match="before its submit"):
        backend.finish_save(_metadata(view, 1, generations=(("req-1", 9),)))


def test_late_finish_of_another_row_cannot_regress_bank_save_event(view) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    backend.submit_save(_metadata(view, 0), _planes(0))
    backend.submit_save(_metadata(view, 2), _planes(0))
    latest_event = ops.calls[-1][2]["done_event"]
    backend.finish_save(_metadata(view, 2))
    backend.finish_save(_metadata(view, 0))
    backend.submit_load(_metadata(view, 3))
    assert ops.calls[-1][2]["wait_event"] is latest_event


def test_finish_checks_batch_identity_and_retains_record_on_publish_failure(
    view,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    metadata = _metadata(view, 0, generations=(("req-1", 7), ("req-2", 8)))
    backend.submit_save(metadata, _planes(0))
    with pytest.raises(ValueError, match="identity"):
        backend.finish_save(_metadata(view, 0))
    publish = Mock(side_effect=[RuntimeError("publish failed"), None])
    monkeypatch.setattr(ops, "finish_publish", publish)
    with pytest.raises(RuntimeError, match="publish failed"):
        backend.finish_save(metadata)
    backend.finish_save(metadata)
    assert publish.call_count == 2
    assert publish.call_args_list[0] == publish.call_args_list[1]


def test_release_allows_newer_generation_without_dropping_physical_events(view) -> None:
    ops = FakeDeviceOps(source_banks=1)
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    old = _metadata(view, 0)
    backend.submit_save(old, _planes(0))
    saved_event = ops.calls[-1][2]["done_event"]
    backend.submit_load(old)
    backend.abort_request("req-1")
    backend.abort_request("req-1")
    calls = ops.calls.copy()

    for generation in (6, 7):
        stale = _metadata(view, 0, generations=(("req-1", generation),))
        with pytest.raises(ValueError, match="released"):
            backend.wait_for_load(stale)
        with pytest.raises(ValueError, match="released"):
            backend.submit_save(stale, _planes(0))
        backend.finish_save(stale)
    assert ops.calls == calls
    new = _metadata(view, 0, generations=(("req-1", 8),))
    backend.wait_for_load(new)
    # Even an unpublished old D2H remains a physical bank dependency.
    backend.submit_load(_metadata(view, 1, generations=(("req-1", 8),)))
    loads = [call for call in ops.calls if call[0] == "submit_load"]
    assert loads[-2][2]["wait_event"] is saved_event
    backend.submit_save(new, _planes(0))
    assert ops.calls[-1][2]["wait_event"] is saved_event
    backend.finish_save(new)


def _coordinator(
    backend: LayerwisePrefillNPUWindowBackend,
) -> LayerwisePrefillWindowCoordinator:
    topology = _topology()
    cache = _DSAKVTopologyCache(
        descriptor=topology,
        layer_name_to_row={
            row.layer_name: row for rows in topology.rows_by_group for row in rows
        },
        execution_to_entry={
            entry.execution_ordinal: entry for entry in topology.executions
        },
        group_layer_names=tuple(
            tuple(row.layer_name for row in rows) for rows in topology.rows_by_group
        ),
        group_cardinalities=(79, 22),
    )
    return LayerwisePrefillWindowCoordinator(cache, backend)


@pytest.mark.parametrize("window", [False, True])
def test_coordinator_npu_wrapper_restores_every_row_across_chunks(
    view,
    window: bool,
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    coordinator = _coordinator(backend)
    for _ in range(2):
        for execution in view.executions:
            callbacks = [_metadata(view, execution[0])]
            if execution[2] is not None:
                callbacks.append(_metadata(view, execution[0], row_kind="indexer"))
            for metadata in callbacks:
                coordinator.wait_for_load(metadata)
                coordinator.wait_for_load(metadata)
            for metadata in callbacks:
                if window:
                    coordinator.submit_save(metadata, _planes(metadata.row.kv_group))
                else:
                    coordinator.save(metadata, _planes(metadata.row.kv_group))
            if window:
                coordinator.submit_load(callbacks[0])
                for metadata in callbacks:
                    coordinator.finish_save(metadata)
                    coordinator.finish_save(metadata)
        assert coordinator.request_persist_done("req-1") is True
    waits = [call for call in ops.calls if call[0] == "wait_for_load"]
    assert sum(call[1][0] == 0 for call in waits) == 2 * 79
    assert sum(call[1][0] == 1 for call in waits) == 2 * 22
    saves = [
        call
        for call in ops.calls
        if call[0] == ("finish_publish" if window else "sync_save")
    ]
    assert len(saves) == 2 * (79 + 22)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_coordinator_npu_wrapper_propagates_failed_restore_and_retries(
    view,
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: bool,
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    coordinator = _coordinator(backend)
    metadata = _metadata(view, 0)
    if not bootstrap:
        coordinator.wait_for_load(metadata)
        coordinator.submit_load(metadata)
        metadata = _metadata(view, 1)
    wait = Mock(side_effect=[RuntimeError("missing restored prefix"), None])
    monkeypatch.setattr(ops, "wait_for_load", wait)
    with pytest.raises(RuntimeError, match="missing restored prefix"):
        coordinator.wait_for_load(metadata)
    coordinator.wait_for_load(metadata)
    coordinator.wait_for_load(metadata)
    assert wait.call_count == 2


def test_coordinator_npu_wrapper_isolates_generations_and_duplicate_publication(
    view,
) -> None:
    future: Future = Future()
    ops = FakeDeviceOps(future=future)
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    coordinator = _coordinator(backend)
    old = _metadata(view, 0)
    coordinator.wait_for_load(old)
    coordinator.submit_save(old, _planes(0))
    coordinator.finish_save(old)
    coordinator.finish_save(old)
    new = _metadata(view, 0, generations=(("req-1", 9),))
    coordinator.wait_for_load(new)
    coordinator.submit_save(new, _planes(0))
    calls = ops.calls.copy()
    for generation in (7, 8):
        stale = _metadata(view, 0, generations=(("req-1", generation),))
        with pytest.raises(RuntimeError, match="superseded"):
            coordinator.wait_for_load(stale)
        coordinator.finish_save(stale)
    assert ops.calls == calls
    coordinator.finish_save(new)
    coordinator.finish_save(new)
    assert coordinator.pending_jobs() == 1
    future.set_result(None)
    coordinator.poll_completed_persists()
    coordinator.finish_save(new)
    assert coordinator.pending_jobs() == 0
    assert len([call for call in ops.calls if call[0] == "finish_publish"]) == 2
    coordinator.release_request("req-1")
    coordinator.finish_save(new)
    newer = _metadata(view, 0, generations=(("req-1", 10),))
    coordinator.wait_for_load(newer)
    coordinator.submit_save(newer, _planes(0))
    coordinator.finish_save(newer)


def test_failed_newer_restore_does_not_skip_older_publication(
    view, monkeypatch
) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    coordinator = _coordinator(backend)
    callbacks = []
    for execution in view.executions:
        callbacks.append(_metadata(view, execution[0]))
        if execution[2] is not None:
            callbacks.append(_metadata(view, execution[0], row_kind="indexer"))
    for metadata in callbacks:
        coordinator.submit_save(metadata, _planes(metadata.row.kv_group))
        if metadata is not callbacks[-1]:
            coordinator.finish_save(metadata)
    assert not coordinator.request_persist_done("req-1")
    monkeypatch.setattr(
        ops, "wait_for_load", Mock(side_effect=RuntimeError("restore failed"))
    )
    with pytest.raises(RuntimeError, match="restore failed"):
        coordinator.wait_for_load(_metadata(view, 0, generations=(("req-1", 9),)))
    coordinator.finish_save(callbacks[-1])
    assert coordinator.request_persist_done("req-1")
    assert len([call for call in ops.calls if call[0] == "finish_publish"]) == 101
    assert not any(backend._submitted.values())


def test_exact_stale_completion_retires_old_record_but_not_bank_event(view) -> None:
    ops = FakeDeviceOps(source_banks=1)
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    coordinator = _coordinator(backend)
    old = _metadata(view, 0)
    new = _metadata(view, 0, generations=(("req-1", 9),))
    coordinator.submit_save(old, _planes(0))
    old_event = ops.calls[-1][2]["done_event"]
    coordinator.wait_for_load(new)
    coordinator.finish_save(old)
    assert coordinator.pending_jobs() == 0
    assert not coordinator.request_persist_done("req-1")
    coordinator.submit_save(new, _planes(0))
    assert ops.calls[-1][2]["wait_event"] is old_event
    assert not [call for call in ops.calls if call[0] == "finish_publish"]
    coordinator.finish_save(new)
    assert len([call for call in ops.calls if call[0] == "finish_publish"]) == 1


@pytest.mark.parametrize("conflict", ["request", "subset", "row"])
def test_wait_cannot_consume_another_load_identity(view, conflict) -> None:
    ops = FakeDeviceOps()
    backend = LayerwisePrefillNPUWindowBackend(view, ops, event_factory=FakeEvent)
    generations = (("req-1", 7), ("req-2", 7))
    backend.submit_load(_metadata(view, 0, generations=generations))
    ledger = backend._ledgers[0][1]
    event = ledger.load_done
    identity = ledger.load_identity
    calls = ops.calls.copy()
    wrong_generations = (
        (("unrelated", 7),)
        if conflict == "request"
        else (("req-1", 7),)
        if conflict == "subset"
        else generations
    )
    with pytest.raises(ValueError, match="does not own"):
        backend.wait_for_load(
            _metadata(
                view,
                3 if conflict == "row" else 1,
                generations=wrong_generations,
            )
        )
    assert ops.calls == calls
    assert ledger.load_done is event
    assert ledger.load_identity == identity
    assert "unrelated" not in backend._active_generations
    backend.wait_for_load(_metadata(view, 1, generations=generations))
    assert ledger.load_identity is None
    assert ledger.load_done is None
