# SPDX-License-Identifier: Apache-2.0
"""Stage 4 NPU transfer-window backend tests."""

# Standard
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.dsa_kv_topology import validate_dsa_kv_topology
from lmcache_ascend.v1.layerwise_prefill_window import (
    LayerwisePrefillDeviceOps,
    LayerwisePrefillNPUWindowBackend,
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


def test_wait_for_load_only_waits_for_submitted_loads(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)

    backend.wait_for_load(_metadata(view, 0))
    assert not [call for call in ops.calls if call[0] == "wait_for_load"]

    backend.submit_load(_metadata(view, 0))
    backend.wait_for_load(_metadata(view, 1))

    waits = [call for call in ops.calls if call[0] == "wait_for_load"]
    assert len(waits) == 1
    assert waits[0][1][:3] == (0, 1, 1)
    # The ledger is consumed; a second wait for the same row no-ops.
    backend.wait_for_load(_metadata(view, 1))
    assert len([call for call in ops.calls if call[0] == "wait_for_load"]) == 1


def test_finish_before_submit_fails_closed(view) -> None:
    backend = _backend(view, FakeDeviceOps(), [FakeEvent()])
    with pytest.raises(ValueError, match="before its submit"):
        backend.finish_save(_metadata(view, 0))


def test_duplicate_submit_fails_closed(view) -> None:
    backend = _backend(view, FakeDeviceOps(), [FakeEvent(), FakeEvent()])
    backend.submit_save(_metadata(view, 0), _planes(0))
    with pytest.raises(ValueError, match="already submitted"):
        backend.submit_save(_metadata(view, 0), _planes(0))


def test_stale_generation_finish_never_publishes(view) -> None:
    ops = FakeDeviceOps()
    events = [FakeEvent() for _ in range(8)]
    backend = _backend(view, ops, events)

    backend.submit_save(_metadata(view, 0, generations=(("req-1", 7),)), _planes(0))
    backend.finish_save(_metadata(view, 0, generations=(("req-1", 8),)))

    publishes = [call for call in ops.calls if call[0] == "finish_publish"]
    assert publishes == []
    assert backend._ledgers == {}


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
    backend.abort_request("req-1")
    with pytest.raises(ValueError, match="aborted request"):
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
