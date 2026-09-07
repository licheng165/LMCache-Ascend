# SPDX-License-Identifier: Apache-2.0
"""CPU content tests through the Ascend adapter and real P sync coordinator."""

# Imported pytest fixtures are intentionally shadowed by fixture arguments.
# ruff: noqa: F811

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

# Third Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LayerwisePrefillWindowCoordinator,
    LoadSpec,
    ReqMeta,
    SaveSpec,
)
from lmcache.v1.cache_engine import LayerwiseStoreResult
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillSyncBackend

# Local
from tests.v1.test_layerwise_prefill_sync import (
    _metadata,
    _registry,
    _request,
    _sentinel,
    _slots,
    _view,
    runtime,  # noqa: F401 -- imported fixture must remain visible to pytest
)


def _make_adapter(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = runtime.engine()
    # Do not let the runtime fixture preconfigure the backend on the adapter's
    # behalf: construction must supply the serving config before capability freeze.
    engine.configure_layerwise_prefill_sync(None)
    serving = runtime.serving
    serving.cache_config = SimpleNamespace(block_size=128)
    serving.model_config.get_num_layers = lambda parallel: 78
    serving.kv_transfer_config = SimpleNamespace(
        get_from_extra_config=lambda key, default: default
    )
    serving.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=1)
    serving.num_speculative_tokens = 1

    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter._role = KVConnectorRole.WORKER
    adapter._vllm_config = serving
    adapter._layerwise_prefill_p_node = True
    adapter.config = engine.config
    adapter.kv_role = "kv_producer"
    adapter.device = "cpu"
    adapter.store_async = engine.is_store_async = False
    engine._direct_store_states = {}
    adapter._parent = SimpleNamespace(_connector_metadata=None)
    adapter._parent._get_connector_metadata = (
        lambda: adapter._parent._connector_metadata
    )
    adapter._manager = SimpleNamespace(
        lmcache_engine=engine, lmcache_engine_metadata=engine.metadata
    )

    # Service startup, metrics and sparse lookup leases are unrelated to P row
    # ownership. Initialize real worker bookkeeping rather than mocking cleanup.
    monkeypatch.setattr(adapter, "_check_legacy_register_kv_caches", Mock())
    monkeypatch.setattr(
        adapter_module.LMCStatsMonitor, "GetOrCreate", Mock(return_value=Mock())
    )
    monkeypatch.setattr(engine, "lookup_unpin", Mock())
    monkeypatch.setattr(engine, "release_shared_cpu_sparse_request", Mock())
    adapter._init_connector_state(adapter._role, serving, engine.config)

    view = _view()
    topology = SimpleNamespace(
        signature=view.signature,
        rows_by_group=tuple(
            tuple(_metadata(view, row, []).row for row in rows)
            for rows in view.rows_by_group
        ),
        executions=tuple(
            _metadata(view, latent, []).execution for _, latent, _ in view.executions
        ),
    )
    adapter._dsa_kv_topology_cache = adapter._build_dsa_kv_topology_cache(
        True,
        SimpleNamespace(
            dsa_kv_topology=topology,
            kv_cache_groups=tuple(
                SimpleNamespace(layer_names=[row[0] for row in rows])
                for rows in view.rows_by_group
            ),
        ),
    )
    adapter._layerwise_prefill_window = adapter._build_layerwise_prefill_window()
    adapter.kv_caches = _registry(engine)
    assert isinstance(adapter._layerwise_prefill_backend, LayerwisePrefillSyncBackend)
    assert isinstance(
        adapter._layerwise_prefill_window, LayerwisePrefillWindowCoordinator
    )
    assert adapter.supports_layerwise_prefill_eager_callbacks
    assert adapter.supports_dsa_index_lmcache
    assert not adapter.supports_layerwise_prefill_transfer_window
    return adapter


@pytest.fixture
def adapter_runtime(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    adapter = _make_adapter(runtime, monkeypatch)
    engine = adapter.lmcache_engine
    window = adapter._layerwise_prefill_window
    backend = adapter._layerwise_prefill_backend
    guards = []
    for owner, names in (
        (
            adapter,
            (
                "save_kv_layer",
                "wait_for_layer_load",
                "_drain_layerwise_retrievers",
                "_should_defer_latent_save_under_tp",
                "_flush_deferred_latent_store",
                "_finalize_layerwise_storer",
                "_direct_prefill_requests",
                "_submit_direct_prefill_requests",
                "_prepare_direct_store_inputs",
            ),
        ),
        (
            engine,
            (
                "retrieve",
                "retrieve_layer",
                "store",
                "store_layer",
                "store_direct_prefill",
                "direct_prefill_store_enabled",
                "wait_for_pending_sync_stores",
                "wait_for_direct_stores",
            ),
        ),
    ):
        for name in names:
            guard = Mock(side_effect=AssertionError(f"P called legacy {name}"))
            monkeypatch.setattr(owner, name, guard)
            guards.append(guard)

    # Spies preserve the real implementations, including release/refcount work.
    events = Mock()
    for owner, name, label in (
        (backend, "bind_step", "bind"),
        (backend, "finish_step", "finish"),
        (backend, "abort_request", "abort"),
        (window, "wait_for_request_persist_done", "barrier"),
        (window, "release_request", "release"),
        (adapter, "_record_prefill_save_group_completed", "record"),
        (adapter, "_mark_prefill_committed", "commit"),
    ):
        spy = Mock(wraps=getattr(owner, name))
        monkeypatch.setattr(owner, name, spy)
        events.attach_mock(spy, label)
    yield SimpleNamespace(
        adapter=adapter, engine=engine, window=window, backend=backend, events=events
    )
    for guard in guards:
        guard.assert_not_called()


def _start(
    adapter: Any,
    requests: list,
    *,
    final: bool = False,
    finished: tuple[str, ...] = (),
    resumed: tuple[str, ...] = (),
) -> list:
    adapter._parent._connector_metadata = LMCacheConnectorMetadata(
        requests=[
            ReqMeta(
                req_id=req.request_id,
                token_ids=list(req.token_ids[: req.compute_end]),
                load_spec=LoadSpec(0, req.restore_end, bool(req.restore_end)),
                save_spec=SaveSpec(req.compute_start, True, True, True),
                is_last_prefill=final,
                resumed_from_preemption=req.request_id in resumed,
                request_configs=req.request_configs,
            )
            for req in requests
        ],
        layerwise_prefill_requests=requests,
        layerwise_prefill_finished=set(finished),
    )
    # The worker input batch is deliberately the reverse of scheduler metadata.
    callbacks = []
    attention = {"unrelated": SimpleNamespace()}
    view = _view()
    for _, latent, indexer in view.executions:
        execution_callbacks = tuple(
            _metadata(view, row, list(reversed(requests)))
            for row in (latent, indexer)
            if row is not None
        )
        attention[latent[0]] = SimpleNamespace(
            layerwise_prefill_callback_metadata=execution_callbacks
        )
        callbacks.extend(execution_callbacks)
    adapter.start_load_kv(SimpleNamespace(attn_metadata=attention))
    return callbacks


def _save_rows(adapter: Any, requests: list, callbacks: list) -> None:
    engine = adapter.lmcache_engine
    cpu = engine.storage_manager.local_cpu_backend
    for metadata in callbacks:
        row = metadata.row
        group, ordinal, bank = row.kv_group, row.row_ordinal, row.bank
        planes = adapter.kv_caches[row.layer_name]
        adapter.wait_for_layerwise_prefill_load(metadata)
        for req in requests:
            slots = _slots(req, bank, group, req.compute_end)
            for index, plane in enumerate(planes):
                expected = _sentinel(req, group, ordinal, index, req.compute_end)
                assert torch.equal(
                    plane.view(-1)[slots[: req.restore_end]],
                    expected[: req.restore_end],
                ), (req.request_id, group, ordinal, index, "restored prefix")
                plane.view(-1)[slots[req.compute_start :]] = expected[
                    req.compute_start :
                ]
        adapter.save_layerwise_prefill_kv_layer(metadata, planes)
        # Recycle immediately, before examining persisted CPU bytes. A deferred
        # all-layer saver would now read poison instead of this row's content.
        for plane in planes:
            plane.fill_(-100)
        for req in requests:
            for start, end, key in engine.token_database.process_tokens(
                list(req.token_ids[: req.compute_end]),
                kv_group=group,
                request_configs=req.request_configs,
            ):
                if start < req.compute_start // 256 * 256:
                    continue  # Unchanged chunks may have been evicted from hot_cache.
                obj = cpu.hot_cache[key.with_new_worker_id(0).get_layer(ordinal)]
                expected = torch.cat(
                    [
                        _sentinel(req, group, ordinal, index, req.compute_end)[
                            start:end
                        ]
                        for index in range(len(planes))
                    ]
                )
                assert torch.equal(obj.tensor.view(-1), expected), (
                    req.request_id,
                    group,
                    ordinal,
                    start,
                    end,
                )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_adapter_configures_engine_before_backend_factory(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, enforce_eager: bool
) -> None:
    runtime.serving.model_config.enforce_eager = enforce_eager
    if not enforce_eager:
        with pytest.raises(ValueError, match="requires eager execution"):
            _make_adapter(runtime, monkeypatch)
        return
    adapter = _make_adapter(runtime, monkeypatch)
    assert adapter.num_layers == 79
    assert adapter._layerwise_prefill_backend.topology_signature == _view().signature
    with pytest.raises(ValueError, match="Cannot reconfigure a frozen"):
        adapter.lmcache_engine.configure_layerwise_prefill_sync(runtime.serving)


@pytest.mark.parametrize("request_count", [1, 4])
def test_adapter_two_chunks_restore_and_commit_all_79_22_rows(
    adapter_runtime: Any, request_count: int
) -> None:
    adapter, engine, events = (
        adapter_runtime.adapter,
        adapter_runtime.engine,
        adapter_runtime.events,
    )
    requests = [_request(index) for index in range(request_count)]
    old = []
    cpu = engine.storage_manager.local_cpu_backend
    for start, end in ((0, 300), (300, 530)):
        if start:
            requests = [
                replace(
                    req,
                    compute_start=start,
                    compute_end=end,
                    restore_end=start,
                    allocation_generation=2,
                )
                for req in requests
            ]
        callbacks = _start(adapter, requests, final=bool(start))
        assert events.bind.call_args.args[0] == list(reversed(requests))
        assert events.bind.call_args.args[1] is adapter.kv_caches
        assert len(callbacks) == 101
        assert [(cb.row.kv_group, cb.row.row_ordinal) for cb in callbacks[-2:]] == [
            (0, 78),
            (1, 21),
        ]
        assert all(cb.execution.execution_ordinal == 78 for cb in callbacks[-2:])
        _save_rows(adapter, requests, callbacks[:-1])
        assert all(
            not adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
        assert adapter.get_completed_decode_window_saves() == {}
        _save_rows(adapter, requests, callbacks[-1:])
        assert all(
            adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
        assert adapter_runtime.window.pending_jobs() == 0
        assert adapter_runtime.window.pending_bytes() == 0
        events.reset_mock()
        adapter.wait_for_save()
        expected_events = [call.finish()]
        for req in adapter._parent._connector_metadata.requests:
            expected_events.append(call.barrier(req.req_id))
            expected_events.extend(
                call.record(
                    req,
                    group,
                    LayerwiseStoreResult(
                        request_id=req.req_id, kv_group=group, committed_end=end
                    ),
                )
                for group in (0, 1)
            )
            expected_events.append(call.commit(req))
        assert events.mock_calls == expected_events
        assert adapter.get_completed_decode_window_saves() == (
            {req.request_id: 512 for req in requests} if start else {}
        )
        assert not adapter._prefill_save_completed_groups
        assert all(
            obj.get_ref_count() == 2 and obj.metadata.pin_count == 1
            for obj in cpu.hot_cache.values()
        )
        if not start:
            old = list(cpu.hot_cache.values())
            assert len(old) == request_count * 101 * 2
            for key in list(cpu.hot_cache):
                cpu.remove(key, force=True)
            assert all(obj.get_ref_count() == 1 for obj in old)

    calls = engine.gpu_connector.calls
    assert sum(direction for direction, *_ in calls) == request_count * 202
    assert sum(not direction for direction, *_ in calls) == request_count * 101
    assert all(
        (starts, ends) == ((0, 256), (256, 300))
        for direction, _, starts, ends in calls
        if not direction
    )
    successor_calls = calls[request_count * 101 :]
    assert all(
        (starts, ends) == ((256, 512), (512, 530))
        for direction, _, starts, ends in successor_calls
        if direction
    )
    assert sum(obj.is_valid() for obj in old) == request_count * 101
    assert adapter.get_finished({req.request_id for req in requests}) == (None, None)
    assert all(not obj.is_valid() for obj in old)
    assert all(
        obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
        for obj in cpu.hot_cache.values()
    )
    assert all(
        not adapter_runtime.window.has_request(req.request_id) for req in requests
    )


@pytest.mark.parametrize("missing_group", [0, 1])
def test_missing_last_mtp_row_never_publishes_committed_success(
    adapter_runtime: Any, missing_group: int
) -> None:
    adapter, events = adapter_runtime.adapter, adapter_runtime.events
    requests = [_request()]
    callbacks = _start(adapter, requests, final=True)
    _save_rows(
        adapter,
        requests,
        [
            cb
            for cb in callbacks
            if not (cb.row.execution_ordinal == 78 and cb.row.kv_group == missing_group)
        ],
    )
    assert not adapter.layerwise_prefill_request_persist_done("req-0")
    events.reset_mock()
    with pytest.raises(ValueError, match="Incomplete layerwise-prefill step.*79/22"):
        adapter.wait_for_save()
    events.finish.assert_called_once_with()
    events.barrier.assert_not_called()
    events.record.assert_not_called()
    events.commit.assert_not_called()
    assert adapter.get_completed_decode_window_saves() == {}
    assert not adapter._prefill_save_completed_groups
    assert not adapter.layerwise_prefill_request_persist_done("req-0")


@pytest.mark.parametrize("retirement", ["finished", "resumed", "both"])
def test_same_id_replacement_releases_old_before_bind_not_new_on_get_finished(
    adapter_runtime: Any, retirement: str
) -> None:
    adapter, events = adapter_runtime.adapter, adapter_runtime.events
    old_request = _request()
    _save_rows(adapter, [old_request], _start(adapter, [old_request], final=True))
    adapter.wait_for_save()
    assert adapter.get_completed_decode_window_saves() == {"req-0": 256}
    cpu = adapter_runtime.engine.storage_manager.local_cpu_backend
    old_objects = list(cpu.hot_cache.values())
    assert len(old_objects) == 202
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)

    # Different tokens and blocks make an unretired continuation invalid, even
    # though the scheduler reuses the same request ID.
    replacement = _request(1, request_id="req-0", generation=2)
    events.reset_mock()
    callbacks = _start(
        adapter,
        [replacement],
        final=True,
        finished=("req-0",) if retirement in ("finished", "both") else (),
        resumed=("req-0",) if retirement in ("resumed", "both") else (),
    )
    expected = [call.barrier("req-0")] if retirement == "resumed" else []
    assert events.mock_calls == expected + [
        call.release("req-0"),
        call.abort("req-0"),
        call.bind([replacement], adapter.kv_caches),
    ]
    assert all(not obj.is_valid() for obj in old_objects)
    _save_rows(adapter, [replacement], callbacks[:1])
    new_objects = list(cpu.hot_cache.values())
    assert len(new_objects) == 2
    events.reset_mock()
    if retirement != "resumed":
        assert adapter.get_finished({"req-0"}) == (None, None)
        assert not adapter._finished_req_ids_waiting_for_save
    assert adapter_runtime.window.has_request("req-0")
    assert all(
        obj.is_valid() and obj.get_ref_count() == 2 and obj.metadata.pin_count == 1
        for obj in new_objects
    )
    _save_rows(adapter, [replacement], callbacks[1:])
    adapter.wait_for_save()
    if retirement != "resumed":
        assert adapter.get_finished({"req-0"}) == (None, None)
    events.release.assert_not_called()
    events.abort.assert_not_called()
    assert adapter.layerwise_prefill_request_persist_done("req-0")
    assert adapter.get_completed_decode_window_saves() == {"req-0": 256}
    assert all(obj.metadata.pin_count == 1 for obj in new_objects)

    # A later, genuine finish (without a replacement in this metadata) releases it.
    adapter._parent._connector_metadata = LMCacheConnectorMetadata(
        layerwise_prefill_requests=[]
    )
    assert adapter.get_finished({"req-0"}) == (None, None)
    events.release.assert_called_once_with("req-0")
    events.abort.assert_called_once_with("req-0")
    assert not adapter_runtime.window.has_request("req-0")
    assert all(
        obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
        for obj in cpu.hot_cache.values()
    )
