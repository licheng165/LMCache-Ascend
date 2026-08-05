# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)

# Local
from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec, ReqMeta
from lmcache_tests.v1.test_dsa_command_protocol import (
    _command,
    _connector,
)
from vllm.v1.core.sched.dsa_types import DSASourceLease, RequestKey


class _Stream:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self.fail:
            raise RuntimeError("stream fence failed")


class _FailedFuture:
    def result(self, timeout=None):
        raise RuntimeError("remote store future failed")


def _ascend_connector(monkeypatch, *, failing_stream: bool = False):
    command = _command(RequestKey("scheduler", "req", 1), "store-ascend")
    base, _, engine = _connector(command)
    connector = object.__new__(LMCacheAscendConnectorV1Impl)
    connector.__dict__.update(base.__dict__)
    connector.store_async = False
    current_stream = _Stream(fail=failing_stream)
    store_stream = _Stream()
    load_stream = _Stream()
    engine.gpu_connector = SimpleNamespace(
        store_stream=store_stream,
        load_stream=load_stream,
    )
    engine.wait_for_pending_sync_stores = lambda: None

    # Third Party
    import lmcache_ascend.integration.vllm.vllm_v1_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module.torch,
        "npu",
        SimpleNamespace(current_stream=lambda: current_stream),
        raising=False,
    )
    return connector, engine, current_stream, store_stream, load_stream


def test_ascend_success_receipts_follow_all_final_stream_fences(monkeypatch) -> None:
    connector, _, current_stream, store_stream, load_stream = _ascend_connector(
        monkeypatch
    )

    connector.save_kv_layer("layer0", None, None)
    assert connector.get_dsa_operation_receipts() == ()
    connector.wait_for_save()
    receipts = connector.get_dsa_operation_receipts()

    assert all(receipt.status == "complete" for receipt in receipts)
    assert current_stream.synchronize_count == 1
    assert store_stream.synchronize_count == 1
    assert load_stream.synchronize_count == 1


def test_ascend_final_stream_fence_failure_emits_failed_receipts(monkeypatch) -> None:
    connector, _, _, _, _ = _ascend_connector(
        monkeypatch,
        failing_stream=True,
    )

    connector.save_kv_layer("layer0", None, None)
    connector.wait_for_save()
    receipts = connector.get_dsa_operation_receipts()

    assert len(receipts) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(receipt.error_code == "final_store_fence_failed" for receipt in receipts)


def test_ascend_backend_future_failure_emits_failed_receipts(monkeypatch) -> None:
    connector, engine, _, _, _ = _ascend_connector(monkeypatch)
    original_store_layer = engine.store_layer

    def store_layer(token_ids, **kwargs):
        original = original_store_layer(token_ids, **kwargs)

        def storer():
            yield next(original)
            result = next(original)
            result.required_futures.append(_FailedFuture())
            yield result

        return storer()

    engine.store_layer = store_layer
    connector.save_kv_layer("layer0", None, None)
    connector.wait_for_save()
    receipts = connector.get_dsa_operation_receipts()

    assert len(receipts) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(receipt.error_code == "backend_future_failed" for receipt in receipts)


def _activate_source_and_bind_lease(connector, lease_id: str):
    key = RequestKey("scheduler", "req", 1)
    connector.save_kv_layer("layer0", None, None)
    connector.wait_for_save()
    connector.get_dsa_operation_receipts()
    metadata = connector._parent._connector_metadata
    activation = _command(
        key,
        "activation-ascend",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-ascend",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)
    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    connector.get_dsa_operation_receipts()
    lease = DSASourceLease(
        source_lease_id=lease_id,
        request_key=key,
        execution_seq=2,
        route_epoch=1,
        source_generation_id="generation-1",
    )
    request = ReqMeta(
        req_id="req",
        token_ids=[0, 1, 2, 3],
        is_sparse_decode=True,
        load_spec=LoadSpec(0, 4, True),
        dsa_request_key=key,
        dsa_route_epoch=1,
        dsa_active_generation_id="generation-1",
        dsa_active_token_prefix_digest=activation.token_prefix_digest,
        dsa_source_lease=lease,
    )
    connector._bind_dsa_active_source(request)
    return request, lease


def test_ascend_source_lease_release_follows_load_stream_fence(monkeypatch) -> None:
    connector, _, current_stream, store_stream, load_stream = _ascend_connector(
        monkeypatch
    )
    request, lease = _activate_source_and_bind_lease(connector, "lease-success")
    counts_before_release = (
        current_stream.synchronize_count,
        store_stream.synchronize_count,
        load_stream.synchronize_count,
    )

    connector._release_dsa_source_leases((request,))

    assert connector.get_released_dsa_source_leases() == (lease,)
    assert (
        current_stream.synchronize_count,
        store_stream.synchronize_count,
        load_stream.synchronize_count,
    ) == tuple(count + 1 for count in counts_before_release)


def test_ascend_failed_source_use_fence_keeps_lease_bound(monkeypatch) -> None:
    connector, _, current_stream, _, _ = _ascend_connector(monkeypatch)
    request, lease = _activate_source_and_bind_lease(connector, "lease-failed")
    current_stream.fail = True

    with pytest.raises(RuntimeError, match="stream fence failed"):
        connector._release_dsa_source_leases((request,))

    assert lease.source_lease_id in connector._dsa_source_lease_bindings
    assert connector.get_released_dsa_source_leases() == ()


def test_ascend_preemption_releases_bound_source_lease(monkeypatch) -> None:
    connector, _, _, _, _ = _ascend_connector(monkeypatch)
    _, lease = _activate_source_and_bind_lease(connector, "lease-preempted")

    connector.handle_preemptions({"req"})

    assert connector.get_released_dsa_source_leases() == (lease,)
    assert lease.source_lease_id not in connector._dsa_source_lease_bindings
    assert lease.source_lease_id not in connector._dsa_bound_source_leases
    assert not connector._dsa_sealed_sources
    assert "req" not in connector._worker_retrieve_state
