# SPDX-License-Identifier: Apache-2.0
"""Opt-in factory validation and real adapter binding with CPU row transport."""

# ruff: noqa: F811

# Standard
from dataclasses import replace
from threading import RLock
from typing import Any
from unittest.mock import Mock

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import LayerwisePrefillWindowCoordinator
from vllm.config import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    LayerwisePrefillCallbackMetadata,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
import pytest
import torch

# First Party
from lmcache_ascend.v1.layerwise_prefill_async import LayerwisePrefillAsyncBackend
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillSyncBackend

# Local
from tests.v1.test_layerwise_prefill_async import (
    DeferredCPUConnector,
    _compute,
    _join_publications,
)
from tests.v1.test_layerwise_prefill_async_adapter import _start
from tests.v1.test_layerwise_prefill_sync import (
    CPUConnector,
    _registry,
    _request,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_adapter import _make_adapter
from tests.v1.test_layerwise_prefill_sync_pages import (
    _assert_remote,
    page_runtime,  # noqa: F401
)


class CustomScheduler(Scheduler):
    pass


def _configure_async(engine: Any, serving: Any) -> None:
    engine.config.extra_config["layerwise_prefill_transfer_window"] = True
    engine.gpu_connector = DeferredCPUConnector()
    serving.scheduler_config = SchedulerConfig.default_factory(
        async_scheduling=False, scheduler_cls=Scheduler
    )
    serving.parallel_config.distributed_executor_backend = "mp"
    engine.configure_layerwise_prefill_sync(serving)


@pytest.fixture
def factory_engine(page_runtime: Any) -> Any:
    engine = page_runtime.engine()
    _configure_async(engine, page_runtime.serving)
    manager = engine.storage_manager
    connector = engine.gpu_connector
    engine.storage_manager = None
    try:
        yield engine
        assert not page_runtime.store.puts and not page_runtime.store.gets
        assert not manager.local_cpu_backend.hot_cache
        assert not engine.gpu_connector.calls
        assert not engine.gpu_connector.prepared
        assert not connector.events
    finally:
        engine.storage_manager = manager


@pytest.mark.parametrize(
    "window", [None, False, True], ids=["default", "sync", "async"]
)
@pytest.mark.parametrize("executor", ["mp", "uni"])
def test_root_and_passive_capabilities_freeze_before_storage_ready(
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    window: bool | None,
    executor: str,
) -> None:
    root = page_runtime.engine(size=2)
    passive = page_runtime.engine(1, root, size=2)
    expected = LayerwisePrefillAsyncBackend if window else LayerwisePrefillSyncBackend
    for rank, engine in enumerate((root, passive)):
        page_runtime.thread.rank = rank
        _configure_async(engine, page_runtime.serving)
        page_runtime.serving.parallel_config.distributed_executor_backend = executor
        if window is None:
            engine.config.extra_config.pop("layerwise_prefill_transfer_window")
        else:
            engine.config.extra_config["layerwise_prefill_transfer_window"] = window
        # Neither the root manager nor passive shared mapping exists at startup.
        with monkeypatch.context() as startup:
            startup.setattr(engine, "storage_manager", None)
            startup.setattr(engine, "shared_cpu_cache_name", None)
            startup.setattr(
                engine, "shared_cpu_cache_passive_allocator", None, raising=False
            )
            backend = engine.layerwise_prefill_window_backend
            assert type(backend) is expected
            assert engine.layerwise_prefill_window_backend is backend
            assert backend.supports_sync_callbacks is True
            assert backend.persists_indexer_group is True
            assert backend.supports_transfer_window is (window is True)
            assert (backend.layer_count(0), backend.layer_count(1)) == (79, 22)
            if window:
                assert backend.manages_pending_work is True
                assert (
                    backend.pending_jobs()
                    == backend.pending_bytes()
                    == backend.pending_futures()
                    == 0
                )
            with pytest.raises(ValueError, match="Cannot reconfigure a frozen"):
                engine.configure_layerwise_prefill_sync(page_runtime.serving)
        assert not engine.gpu_connector.events
        assert not engine.gpu_connector.prepared
        assert not engine.gpu_connector.layouts
    assert not page_runtime.store.puts and not page_runtime.store.gets
    assert not root.storage_manager.local_cpu_backend.hot_cache


@pytest.mark.parametrize("value", [1, "true"])
def test_factory_requires_a_boolean_opt_in(factory_engine: Any, value: Any) -> None:
    factory_engine.config.extra_config["layerwise_prefill_transfer_window"] = value
    with pytest.raises(ValueError, match="layerwise_prefill_transfer_window.*boolean"):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize("flag", [None, "false"], ids=["unset", "false"])
def test_p_feature_off_does_not_enable_or_validate_async_window(
    factory_engine: Any,
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    flag: str | None,
) -> None:
    if flag is None:
        monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE")
    else:
        monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", flag)
    page_runtime.serving.scheduler_config.async_scheduling = True
    page_runtime.serving.parallel_config.distributed_executor_backend = "ray"
    page_runtime.serving.model_config.enforce_eager = False
    factory_engine.gpu_connector.supports_layerwise_prefill_async_rows = False
    assert factory_engine.layerwise_prefill_window_backend is None


@pytest.mark.parametrize("window", [None, False], ids=["default", "explicit-sync"])
def test_sync_factory_keeps_async_scheduler_default_unaffected(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, window: bool | None
) -> None:
    engine = runtime.engine()
    if window is not None:
        engine.config.extra_config["layerwise_prefill_transfer_window"] = window
    runtime.serving.scheduler_config = SchedulerConfig.default_factory(
        async_scheduling=True
    )
    runtime.serving.parallel_config.distributed_executor_backend = "ray"
    # The sync-only helper also checks that the adapter supplies serving config.
    monkeypatch.setattr(runtime, "engine", lambda: engine)
    adapter = _make_adapter(runtime, monkeypatch)
    assert type(adapter._layerwise_prefill_backend) is LayerwisePrefillSyncBackend
    assert runtime.serving.scheduler_config.async_scheduling is True
    assert not adapter.supports_layerwise_prefill_transfer_window
    assert not engine.gpu_connector.calls


@pytest.mark.parametrize("async_scheduling", [None, False, True])
def test_async_factory_accepts_async_scheduling(
    factory_engine: Any, page_runtime: Any, async_scheduling: Any
) -> None:
    page_runtime.serving.scheduler_config.async_scheduling = async_scheduling
    # vLLM defaults async_scheduling to None and auto-enables it when the
    # executor/speculative configuration is compatible, so all three values
    # must construct the managed window backend without --no-async-scheduling.
    assert (
        type(factory_engine.layerwise_prefill_window_backend)
        is LayerwisePrefillAsyncBackend
    )


@pytest.mark.parametrize(
    "executor", ["ray", "external_launcher", "custom", object, None]
)
def test_async_factory_rejects_unsupported_executors(
    factory_engine: Any, page_runtime: Any, executor: Any
) -> None:
    page_runtime.serving.parallel_config.distributed_executor_backend = executor
    with pytest.raises(ValueError, match="mp/uni executor"):
        _ = factory_engine.layerwise_prefill_window_backend


def test_async_factory_requires_page_first(factory_engine: Any) -> None:
    factory_engine.config.extra_config["mooncake_page_first_multi_buffer"] = False
    with pytest.raises(ValueError, match="requires page-first"):
        _ = factory_engine.layerwise_prefill_window_backend


def test_async_factory_requires_serving_config(factory_engine: Any) -> None:
    factory_engine.configure_layerwise_prefill_sync(None)
    with pytest.raises(ValueError, match="configure_layerwise_prefill_sync"):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        None,
        Scheduler,
        "vllm.v1.core.sched.scheduler.Scheduler",
        AsyncScheduler,
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
    ],
)
def test_async_factory_accepts_resolved_core_scheduler(
    factory_engine: Any, page_runtime: Any, scheduler_cls: Any
) -> None:
    scheduler = page_runtime.serving.scheduler_config
    scheduler.scheduler_cls = scheduler_cls
    assert scheduler.get_scheduler_cls() in (Scheduler, AsyncScheduler)
    assert (
        type(factory_engine.layerwise_prefill_window_backend)
        is LayerwisePrefillAsyncBackend
    )


def test_async_factory_accepts_async_resolution_without_a_configured_class(
    factory_engine: Any, page_runtime: Any
) -> None:
    scheduler = page_runtime.serving.scheduler_config
    scheduler.scheduler_cls = None
    scheduler.async_scheduling = True
    # Production async-scheduling resolution: the lazy import returns the
    # AsyncScheduler subclass without any scheduler_cls override.
    assert scheduler.get_scheduler_cls() is AsyncScheduler
    assert (
        type(factory_engine.layerwise_prefill_window_backend)
        is LayerwisePrefillAsyncBackend
    )


@pytest.mark.parametrize(
    "scheduler_cls", [CustomScheduler, f"{__name__}.CustomScheduler"]
)
def test_async_factory_rejects_custom_scheduler_class_and_path(
    factory_engine: Any, page_runtime: Any, scheduler_cls: Any
) -> None:
    page_runtime.serving.scheduler_config.scheduler_cls = scheduler_cls
    with pytest.raises(ValueError, match="Recompute/custom scheduling"):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize(
    "module,name",
    [
        ("vllm_ascend.core.recompute_scheduler", "RecomputeScheduler"),
        ("custom.scheduler", "Scheduler"),
        ("vllm.v1.core.sched.scheduler", "CustomScheduler"),
    ],
)
def test_async_factory_checks_actual_scheduler_resolver_not_configured_name(
    factory_engine: Any,
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    name: str,
) -> None:
    scheduler = page_runtime.serving.scheduler_config
    assert scheduler.scheduler_cls is Scheduler
    assert scheduler.async_scheduling is False
    # Model Balance-style resolver patches without importing their global patches.
    resolved = Mock(return_value=type(name, (Scheduler,), {"__module__": module}))
    monkeypatch.setattr(scheduler, "get_scheduler_cls", resolved)
    with pytest.raises(ValueError, match="Recompute/custom scheduling"):
        _ = factory_engine.layerwise_prefill_window_backend
    resolved.assert_called_once_with()


@pytest.mark.parametrize("marker", [True, False], ids=["with_marker", "no_marker"])
def test_async_factory_accepts_only_the_real_balance_patch(
    factory_engine: Any,
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    marker: bool,
) -> None:
    scheduler = page_runtime.serving.scheduler_config
    namespace = {"__module__": "vllm_ascend.patch.platform.patch_balance_schedule"}
    if marker:
        namespace["balance_gather"] = lambda self: None
    # The Ascend balance patch rebinds the module-level Scheduler class; it is
    # positively identified by module/qualname plus its balance_gather helper.
    resolved = Mock(return_value=type("BalanceScheduler", (Scheduler,), namespace))
    monkeypatch.setattr(scheduler, "get_scheduler_cls", resolved)
    if marker:
        assert (
            type(factory_engine.layerwise_prefill_window_backend)
            is LayerwisePrefillAsyncBackend
        )
    else:
        with pytest.raises(ValueError, match="Recompute/custom scheduling"):
            _ = factory_engine.layerwise_prefill_window_backend
    resolved.assert_called_once_with()


@pytest.mark.parametrize(
    "hook",
    [
        "prepare_layerwise_prefill_row",
        "submit_layerwise_prefill_row",
        "wait_layerwise_prefill_row",
        "complete_layerwise_prefill_row",
        "drain_layerwise_prefill_transfers",
    ],
)
def test_async_factory_requires_every_row_hook(factory_engine: Any, hook: str) -> None:
    setattr(factory_engine.gpu_connector, hook, None)
    with pytest.raises(ValueError, match="concrete async row hooks"):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize(
    "capability",
    [False, None, 1, "true", pytest.param(Mock(return_value=True), id="dynamic-mock")],
)
def test_async_factory_requires_literal_true_row_capability(
    factory_engine: Any, capability: Any
) -> None:
    factory_engine.gpu_connector.supports_layerwise_prefill_async_rows = capability
    with pytest.raises(ValueError, match="concrete async row hooks"):
        _ = factory_engine.layerwise_prefill_window_backend


def test_async_factory_rejects_capability_without_concrete_hooks(
    factory_engine: Any,
) -> None:
    factory_engine.gpu_connector = CPUConnector()
    factory_engine.gpu_connector.supports_layerwise_prefill_async_rows = True
    with pytest.raises(ValueError, match="concrete async row hooks"):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize(
    "owner,setting,value,message",
    [
        ("model_config", "enforce_eager", False, "requires eager execution"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "enable_dbo", True, "no DBO"),
        ("parallel_config", "tensor_parallel_size", 2, "single-host TP"),
        ("metadata", "local_world_size", 2, "single-host TP"),
        ("metadata", "first_rank", 1, "first_rank=0"),
        ("metadata", "kv_dtype", torch.float16, "BF16 MLA 79/22"),
        ("config", "store_async", True, "store_async=false"),
        ("config", "chunk_size", 128, "chunk_size=256"),
        ("config", "shared_cpu_cache_strict", False, "strict shared LocalCPU"),
        (
            "gpu_connector",
            "transfer_layerwise_prefill_row",
            None,
            "synchronous NPU row hook",
        ),
    ],
)
def test_async_factory_preserves_sync_deployment_prerequisites(
    factory_engine: Any,
    page_runtime: Any,
    owner: str,
    setting: str,
    value: Any,
    message: str,
) -> None:
    parent = page_runtime.serving if owner.endswith("_config") else factory_engine
    setattr(getattr(parent, owner), setting, value)
    with pytest.raises(ValueError, match=message):
        _ = factory_engine.layerwise_prefill_window_backend


@pytest.mark.parametrize("request_count", [1, 4])
def test_factory_to_adapter_binds_actual_101_rows_and_persists_two_steps(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, request_count: int
) -> None:
    adapter = _make_adapter(page_runtime, monkeypatch)
    # The shared helper asserts sync mode. Reuse its worker bookkeeping with a
    # fresh engine, not a replaced frozen backend or a patched factory/validator.
    engine = page_runtime.engine()
    _configure_async(engine, page_runtime.serving)
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
    manager = engine.storage_manager
    with monkeypatch.context() as startup:
        startup.setattr(engine, "storage_manager", None)
        adapter._layerwise_prefill_window = adapter._build_layerwise_prefill_window()
    adapter.kv_caches = _registry(engine)
    backend, window = (
        adapter._layerwise_prefill_backend,
        adapter._layerwise_prefill_window,
    )
    assert type(backend) is LayerwisePrefillAsyncBackend
    assert engine.layerwise_prefill_window_backend is backend
    assert isinstance(window, LayerwisePrefillWindowCoordinator)
    assert window.manages_pending_work
    assert adapter.supports_layerwise_prefill_transfer_window
    assert adapter.supports_layerwise_prefill_eager_callbacks
    assert adapter.supports_dsa_index_lmcache
    assert not engine.gpu_connector.events
    assert not manager.local_cpu_backend.hot_cache
    assert not page_runtime.store.puts and not page_runtime.store.gets

    requests = [_request(i) for i in range(request_count)]
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
        assert len(callbacks) == 101
        assert all(isinstance(m, LayerwisePrefillCallbackMetadata) for m in callbacks)
        assert all(
            backend._callbacks[m.row.kv_group, m.row.row_ordinal] is m
            for m in callbacks
        )
        before = len(page_runtime.store.puts)
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
            _join_publications(backend)
            assert (
                window.pending_jobs()
                == window.pending_bytes()
                == window.pending_futures()
                == 0
            )
        assert len(page_runtime.store.puts) == before
        assert all(
            not adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
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
    assert adapter.get_finished({req.request_id for req in requests}) == (None, None)
    assert all(not window.has_request(req.request_id) for req in requests)
