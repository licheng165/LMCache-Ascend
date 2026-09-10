# SPDX-License-Identifier: Apache-2.0
"""Page-only P persistence through real Mooncake/RemoteBackend and adapter code."""

# ruff: noqa: F811

# Standard
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import asyncio
import ctypes
import multiprocessing
import sys

# Third Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.mooncake_layout import mooncake_page_key, mooncake_payload_layout
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.token_database import ChunkedTokenDatabase
import pytest
import torch
import vllm.distributed.parallel_state

# First Party
from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
    LMCacheAscendConnectorV1Dynamic,
)
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillFenceError
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)

# Local
from tests.v1.test_layerwise_prefill_sync import (
    _metadata,
    _patch_cpu_slot_preparer,
    _registry,
    _request,
    _sentinel,
    _step,
    _view,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_adapter import (
    _make_adapter,
    _make_tp_adapters,
    _save_rows,
    _start,
)


class ByteStore:
    """Native Mooncake substitute: copy raw pointers, never retain input tensors."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.puts: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
        self.gets: list[tuple[str, ...]] = []
        self.submitted = Event()
        self.gate = Event()
        self.gate.set()
        self.fail = False

    def setup(self, *args: Any) -> int:
        return 0

    def register_buffer(self, *args: Any) -> int:
        return 0

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [int(key in self.data) for key in keys]

    def batch_put_from_multi_buffers(
        self, keys: list, pointers: list, sizes: list, replica: Any
    ) -> list[int]:
        self.submitted.set()
        assert self.gate.wait(10)
        self.puts.append(("page", tuple(keys), tuple(map(len, pointers))))
        if self.fail:
            return [-1] * len(keys)
        for key, ptrs, lengths in zip(keys, pointers, sizes, strict=True):
            self.data[key] = b"".join(
                ctypes.string_at(ptr, size)
                for ptr, size in zip(ptrs, lengths, strict=True)
            )
        return [0] * len(keys)

    def batch_put_from(
        self, keys: list, pointers: list, sizes: list, replica: Any
    ) -> list[int]:
        self.puts.append(("tail", tuple(keys), (1,) * len(keys)))
        for key, ptr, size in zip(keys, pointers, sizes, strict=True):
            self.data[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)

    def batch_get_into_multi_buffers(
        self, keys: list, pointers: list, sizes: list
    ) -> list[int]:
        self.gets.append(tuple(keys))
        # This executes on a remote I/O thread. Reclamation must not be blocked
        # by a protect_pins scope held by the synchronous caller.
        PinMonitor.GetOrCreate()._check_timeouts()
        with PinMonitor.GetOrCreate().protect_pins():
            pass
        result = []
        for key, ptrs, lengths in zip(keys, pointers, sizes, strict=True):
            data = self.data.get(key)
            if data is None or len(data) != sum(lengths):
                result.append(-1)
                continue
            offset = 0
            for ptr, size in zip(ptrs, lengths, strict=True):
                ctypes.memmove(ptr, data[offset : offset + size], size)
                offset += size
            result.append(offset)
        return result

    def batch_get_into(self, keys: list, pointers: list, sizes: list) -> list[int]:
        self.gets.append(tuple(keys))
        result = []
        for key, ptr, size in zip(keys, pointers, sizes, strict=True):
            data = self.data.get(key)
            if data is None or len(data) > size:
                result.append(-1)
                continue
            ctypes.memmove(ptr, data, len(data))
            result.append(len(data))
        return result


@pytest.fixture
def page_runtime(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    store = ByteStore()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        SimpleNamespace(
            MooncakeDistributedStore=lambda: store, ReplicateConfig=SimpleNamespace
        ),
    )
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    make_engine = runtime.engine

    def engine(
        *args: Any, raw_dims: dict[int, int] | None = None, **kwargs: Any
    ) -> Any:
        result = make_engine(*args, **kwargs)
        config = result.config
        config.remote_url = "mooncakestore://fake:50051"
        config.remote_serde = "naive"
        config.extra_config.update(
            mooncake_page_first_multi_buffer=True,
            save_chunk_meta=False,
            mooncake_dsa_raw_token_dims={0: 2, 1: 1} if raw_dims is None else raw_dims,
            local_hostname="fake",
            metadata_server="fake",
            master_server_address="fake:50051",
            transfer_timeout=10,
        )
        result.metadata.runtime_kv_group_layer_counts = (79, 22)
        result.token_database = ChunkedTokenDatabase(config, result.metadata)
        assert (
            result.token_database.mooncake_payload_layout
            == mooncake_payload_layout(config, result.metadata)[0]
        )
        if result.metadata.is_first_rank():
            manager = result.storage_manager
            manager.config = config
            cpu = manager.local_cpu_backend
            remote = RemoteBackend(config, result.metadata, loop, cpu, dst_device="cpu")
            assert isinstance(remote.connection, InstrumentedRemoteConnector)
            assert isinstance(remote.connection._connector, MooncakestoreConnector)
            manager.storage_backends["RemoteBackend"] = remote
        return result

    monkeypatch.setattr(runtime, "engine", engine)
    runtime.store = store
    yield runtime
    store.gate.set()
    asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(10)
    loop.close()


def _assert_remote(engine: Any, req: Any, store: ByteStore) -> None:
    for group, count in enumerate((79, 22)):
        for start, end, key in engine.token_database.process_tokens(
            list(req.token_ids[: req.compute_end]),
            kv_group=group,
            request_configs=req.request_configs,
        ):
            assert (
                dict(key.tags)["payload_v2"]
                == mooncake_payload_layout(engine.config, engine.metadata)[0]
            )
            rows = [
                torch.cat(
                    [
                        _sentinel(req, group, row, plane, end)[start:end]
                        for plane in range(2 if group == 0 else 1)
                    ]
                )
                for row in range(count)
            ]
            if end - start == 256:
                assert all(
                    key.get_layer(row).to_string() not in store.data
                    for row in range(count)
                )
                actual = torch.frombuffer(
                    bytearray(store.data[mooncake_page_key(key, count)]),
                    dtype=torch.bfloat16,
                )
                assert torch.equal(actual, torch.cat(rows))
            else:
                for row, expected in enumerate(rows):
                    actual = torch.frombuffer(
                        bytearray(store.data[key.get_layer(row).to_string()]),
                        dtype=torch.bfloat16,
                    )
                    assert torch.equal(actual, expected)


@pytest.mark.parametrize("request_count", [1, 4])
def test_page_adapter_end_only_commit_partial_successor_and_warm_reuse(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, request_count: int
) -> None:
    adapter = _make_adapter(page_runtime, monkeypatch)
    engine, store = adapter.lmcache_engine, page_runtime.store
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
            # Warm load must not query any storage or publish old handles.
            monkeypatch.setattr(
                engine.storage_manager,
                "batched_get",
                Mock(side_effect=AssertionError("warm get")),
            )
            monkeypatch.setattr(
                engine,
                "resolve_layerwise_prefill_group",
                Mock(side_effect=AssertionError("warm group")),
            )
        before = len(store.puts)
        prepared_before = len(engine.gpu_connector.prepared)
        callbacks = _start(adapter, requests, final=bool(start))
        assert len(engine.gpu_connector.prepared) - prepared_before == 4 * request_count
        _save_rows(adapter, requests, callbacks)
        assert len(engine.gpu_connector.prepared) - prepared_before == 4 * request_count
        assert len(store.puts) == before
        assert not store.gets
        assert all(
            not adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
        assert adapter.get_completed_decode_window_saves() == {}
        adapter.wait_for_save()
        assert not adapter._layerwise_prefill_backend._slots
        assert all(ref() is None for ref in engine.gpu_connector.prepared)
        assert all(
            adapter.layerwise_prefill_request_persist_done(req.request_id)
            for req in requests
        )
        assert len(store.puts) - before == request_count * 4
        for req in requests:
            _assert_remote(engine, req, store)
        assert all(
            len(keys) == 1 for kind, keys, _ in store.puts[before:] if kind == "page"
        )
    published = Counter(
        key for kind, keys, _ in store.puts for key in keys if kind == "page"
    )
    assert set(published.values()) == {1}
    assert adapter.get_completed_decode_window_saves() == {
        req.request_id: 512 for req in requests
    }
    adapter.get_finished({req.request_id for req in requests})
    assert all(
        obj.metadata.pin_count == 0 and obj.get_ref_count() == 1
        for obj in engine.storage_manager.local_cpu_backend.hot_cache.values()
    )


@pytest.mark.parametrize("cold", [True, False])
@pytest.mark.parametrize("start,end", [(300, 530), (299, 300), (511, 512)])
def test_page_only_bootstrap_complete_groups_and_full_external_republication(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, cold: bool, start: int, end: int
) -> None:
    engine = page_runtime.engine()
    backend, store = engine.layerwise_prefill_window_backend, page_runtime.store
    restore_end = 512 if start == 511 else 300
    seed = _request(request_id="seed", end=restore_end)
    _step(engine, backend, [seed])
    _assert_remote(engine, seed, store)
    backend.abort_request("seed")
    cpu = engine.storage_manager.local_cpu_backend
    if cold:
        for key in list(cpu.hot_cache):
            cpu.remove(key, force=True)
        assert cpu.memory_allocator.num_active_allocations == 0
        # A newly constructed consumer must generate the same tagged page and
        # tail keys, without sharing the producer's TokenDatabase instance.
        engine.token_database = ChunkedTokenDatabase(engine.config, engine.metadata)
    else:
        # Untrusted LocalCPU may be left by an earlier failed remote put.
        store.data.clear()
    store.puts.clear()
    resolved = Mock(wraps=engine.resolve_layerwise_prefill_group)
    monkeypatch.setattr(engine, "resolve_layerwise_prefill_group", resolved)
    req = _request(request_id="external", start=start, end=end, restore_end=restore_end)
    # Drive the public backend directly; suppress only final commit to inspect it.
    finish = backend.finish_step
    monkeypatch.setattr(backend, "finish_step", lambda: None)
    _step(engine, backend, [req])
    assert resolved.call_count == 2
    assert not store.puts
    assert not backend._pending_rows
    assert backend._step_future is not None and not backend._step_future.done()
    finish()
    _assert_remote(engine, req, store)
    assert sum(len(keys) for kind, keys, _ in store.puts if kind == "page") == 2 * (
        end // 256
    )
    backend.abort_request("external")


@pytest.mark.parametrize("failure", ["success", "remote", "none_future", "commit_ack"])
def test_page_tp_commit_waits_and_failure_never_commits(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = page_runtime.engine(size=2)
    passive = page_runtime.engine(1, root, size=2)
    page_runtime.thread.rank = 0
    ra = _make_adapter(
        SimpleNamespace(engine=lambda: root, serving=page_runtime.serving), monkeypatch
    )
    page_runtime.thread.rank = 1
    pa = _make_adapter(
        SimpleNamespace(engine=lambda: passive, serving=page_runtime.serving),
        monkeypatch,
    )
    rb, pb = ra._layerwise_prefill_backend, pa._layerwise_prefill_backend
    adapters = [ra, pa]
    commits = [Mock(wraps=adapter._mark_prefill_committed) for adapter in adapters]
    for adapter, commit in zip(adapters, commits, strict=True):
        monkeypatch.setattr(adapter, "_mark_prefill_committed", commit)
    requests = [_request()]
    page_runtime.parallel(
        lambda: _save_rows(ra, requests, _start(ra, requests, final=True)),
        lambda: _save_rows(pa, requests, _start(pa, requests, final=True)),
    )
    futures = [rb._step_future, pb._step_future]
    owners = [
        [obj for prefix in backend._prefixes.values() for obj in prefix.objects]
        for backend in (rb, pb)
    ]
    assert all(isinstance(future, Future) and not future.done() for future in futures)
    for engine, backend in ((root, rb), (passive, pb)):
        assert len(engine.gpu_connector.prepared) == len(backend._slots) == 4
    store = page_runtime.store
    assert not store.puts
    assert all(
        not adapter.layerwise_prefill_request_persist_done("req-0")
        for adapter in adapters
    )
    if failure == "none_future":
        monkeypatch.setattr(
            root.storage_manager.storage_backends["RemoteBackend"],
            "batched_submit_put_task",
            lambda *args: None,
        )
    if failure == "commit_ack":
        ack = passive.layerwise_prefill_ack

        def reject(identity: Any, error: Any = None, *, flush: bool = True) -> None:
            if identity[1][0] == "commit":
                error = ValueError("commit ACK failed")
            ack(identity, error, flush=flush)

        monkeypatch.setattr(passive, "layerwise_prefill_ack", reject)
    store.fail = failure == "remote"
    store.gate.clear()

    def finish(index: int) -> None:
        if failure == "success":
            adapters[index].wait_for_save()
        else:
            with pytest.raises(ValueError, match="put|ACK"):
                adapters[index].wait_for_save()

    with ThreadPoolExecutor(max_workers=1) as pool:
        committed = pool.submit(
            page_runtime.parallel, lambda: finish(0), lambda: finish(1)
        )
        if failure != "none_future":
            assert store.submitted.wait(5)
            assert not committed.done()
            assert all(not future.done() for future in futures)
            assert all(
                not adapter.layerwise_prefill_request_persist_done("req-0")
                for adapter in adapters
            )
            for commit in commits:
                commit.assert_not_called()
        store.gate.set()
        committed.result(20)
    assert all(future.done() for future in futures)
    assert not rb._slots and not pb._slots
    assert all(
        (future.exception() is None) == (failure == "success") for future in futures
    )
    for adapter, commit in zip(adapters, commits, strict=True):
        assert adapter.layerwise_prefill_request_persist_done("req-0") == (
            failure == "success"
        )
        if failure == "success":
            commit.assert_called_once()
        else:
            commit.assert_not_called()
            assert adapter.get_completed_decode_window_saves() == {}
    if failure != "success":
        for rank, (adapter, objects) in enumerate(zip(adapters, owners, strict=True)):
            backend = adapter._layerwise_prefill_backend
            assert not backend._prefixes and not backend._pending_rows
            assert not adapter._layerwise_prefill_window.has_request("req-0")
            assert all(
                obj.metadata.pin_count == 0
                and obj.get_ref_count() == (1 if rank == 0 else 0)
                for obj in objects
            )


@pytest.mark.parametrize("missing", ["page", "tail"])
def test_bootstrap_missing_page_or_tail_releases_pending_owners(
    page_runtime: Any, missing: str
) -> None:
    engine = page_runtime.engine()
    backend, store = engine.layerwise_prefill_window_backend, page_runtime.store
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    cpu = engine.storage_manager.local_cpu_backend
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)
    victim = next(
        key
        for key in store.data
        if key.startswith("__lmcache_page_v1__") == (missing == "page")
    )
    del store.data[victim]
    requests = [_request(start=300, end=530)]
    backend.bind_step(requests, _registry(engine))
    with pytest.raises(ValueError, match="Missing required"):
        backend.wait_for_load(_metadata(_view(), _view().rows_by_group[0][0], requests))
    assert backend._step_future.exception() is not None
    assert not backend._pending_rows
    assert not engine.gpu_connector.calls[101:]
    backend.abort_request("req-0")


def test_pending_group_abort_releases_unpublished_rows(page_runtime: Any) -> None:
    engine = page_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    requests = [_request(start=300, end=530)]
    backend.bind_step(requests, _registry(engine))
    backend.wait_for_load(_metadata(_view(), _view().rows_by_group[0][0], requests))
    assert len(backend._pending_rows) == 78
    assert len(backend._prefixes) == 1
    future = backend._step_future
    backend.abort_request("req-0")
    assert future.exception() is not None
    assert not backend._pending_rows
    assert all(
        obj.metadata.pin_count == 0 and obj.get_ref_count() == 1
        for obj in engine.storage_manager.local_cpu_backend.hot_cache.values()
    )


@pytest.mark.parametrize("size", [1, 2])
def test_page_capabilities_precede_storage_and_kv_registration(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    root = page_runtime.engine(size=size)
    engines = [root]
    if size == 2:
        engines.append(page_runtime.engine(1, root, size=size))
    adapters = []
    for rank, engine in enumerate(engines):
        page_runtime.thread.rank = rank
        engine.num_layers = engine.metadata.kv_shape[0]
        manager, name = engine.storage_manager, engine.shared_cpu_cache_name
        engine.storage_manager, engine.shared_cpu_cache_name = None, None
        try:
            # Production constructs the adapter/backend before register_kv_caches
            # calls manager.post_init(). No transport/slab is available yet.
            adapter = _make_adapter(
                SimpleNamespace(
                    engine=lambda engine=engine: engine, serving=page_runtime.serving
                ),
                monkeypatch,
            )
            connector = object.__new__(LMCacheAscendConnectorV1Dynamic)
            connector._lmcache_engine = adapter
            assert connector.supports_layerwise_prefill_p_node
            assert connector.supports_layerwise_prefill_eager_callbacks
            assert connector.supports_dsa_index_lmcache
            assert not connector.supports_layerwise_prefill_transfer_window
            assert engine.storage_manager is None
            assert not engine.gpu_connector.layouts

            def post_init(
                engine: Any = engine, manager: Any = manager, name: str = name
            ) -> None:
                assert engine.storage_manager is None
                assert [
                    group.num_layers
                    for group in engine.metadata.kv_layer_groups_manager.kv_layer_groups
                ] == [79, 22]
                engine.storage_manager, engine.shared_cpu_cache_name = manager, name

            adapter._manager.post_init = Mock(side_effect=post_init)
            adapter.dsa_kv_topology = adapter._dsa_kv_topology_cache.descriptor
            adapter.kv_caches = {}
            connector.register_kv_caches(
                {name: tuple(planes) for name, planes in _registry(engine).items()}
            )
            adapter._manager.post_init.assert_called_once_with()
            adapters.append(adapter)
        finally:
            engine.storage_manager, engine.shared_cpu_cache_name = manager, name

    connection = root.storage_manager.storage_backends["RemoteBackend"].connection
    validate = Mock(wraps=connection.validate_page_first_layout)
    monkeypatch.setattr(connection, "validate_page_first_layout", validate)
    req = _request()

    def run(adapter: Any) -> None:
        _save_rows(adapter, [req], _start(adapter, [req], final=True))
        adapter.wait_for_save()
        assert adapter.layerwise_prefill_request_persist_done(req.request_id)

    page_runtime.parallel(
        *(lambda adapter=adapter: run(adapter) for adapter in adapters)
    )
    assert [call.args[:2] for call in validate.call_args_list] == [(0, 79), (1, 22)]
    _assert_remote(root, req, page_runtime.store)


@pytest.mark.parametrize(
    "failure", ["storage", "remote", "connection", "capability", "layout_hook"]
)
def test_page_transport_readiness_fails_on_all_ranks_before_transfer(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    root = adapters[0].lmcache_engine
    manager = root.storage_manager
    remote = manager.storage_backends["RemoteBackend"]
    if failure == "storage":
        root.storage_manager = None
    elif failure == "remote":
        monkeypatch.delitem(manager.storage_backends, "RemoteBackend")
    elif failure == "connection":
        monkeypatch.setattr(remote, "connection", None)
    elif failure == "capability":
        monkeypatch.setattr(page_runtime.store, "batch_get_into_multi_buffers", None)
    else:
        monkeypatch.setattr(remote.connection, "validate_page_first_layout", None)

    def bind(adapter: Any) -> None:
        with pytest.raises(ValueError, match="requires.*(storage|RemoteBackend)"):
            _start(adapter, [_request()])
        assert not adapter.layerwise_prefill_request_persist_done("req-0")
        assert not adapter.lmcache_engine.gpu_connector.calls

    try:
        page_runtime.parallel(
            *(lambda adapter=adapter: bind(adapter) for adapter in adapters)
        )
        assert not page_runtime.store.gets and not page_runtime.store.puts
    finally:
        root.storage_manager = manager


@pytest.mark.parametrize("setting", ["url", "serde", "metadata", "merged", "direct"])
def test_page_factory_rejects_invalid_transport(
    page_runtime: Any, setting: str
) -> None:
    engine = page_runtime.engine()
    if setting == "url":
        engine.config.remote_url = "redis://fake"
    elif setting == "serde":
        engine.config.remote_serde = "cachegen"
    elif setting == "metadata":
        engine.config.extra_config["save_chunk_meta"] = True
    elif setting == "merged":
        engine.config.extra_config["mooncake_layer_merged_page_objects"] = True
    elif setting == "direct":
        engine.config.extra_config["mooncake_direct_npu_prefill_store"] = True
    with pytest.raises(ValueError, match="requires|support"):
        _ = engine.layerwise_prefill_window_backend


def test_tp_bind_rejects_page_mode_divergence(page_runtime: Any) -> None:
    root = page_runtime.engine(size=2)
    passive = page_runtime.engine(1, root, size=2)
    passive.config.extra_config["mooncake_page_first_multi_buffer"] = False
    page_runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    page_runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend

    def bind(engine: Any, backend: Any) -> None:
        with pytest.raises(ValueError, match="identity mismatch"):
            backend.bind_step([_request()], _registry(engine))

    page_runtime.parallel(lambda: bind(root, rb), lambda: bind(passive, pb))
    assert not root.gpu_connector.calls and not passive.gpu_connector.calls


def test_page_batches_bound_chunks_not_layers(page_runtime: Any) -> None:
    engine = page_runtime.engine(blocks=80, slab_bytes=16 << 20)
    backend, store = engine.layerwise_prefill_window_backend, page_runtime.store
    end = 17 * 256
    req = replace(
        _request(),
        token_ids=tuple(range(end)),
        compute_end=end,
        block_ids_by_bank=tuple(
            (tuple(range(1 + bank * 34, 35 + bank * 34)),) * 2 for bank in (0, 1)
        ),
    )
    _step(engine, backend, [req])
    assert [(len(keys), set(counts)) for _, keys, counts in store.puts] == [
        (16, {79}),
        (1, {79}),
        (16, {22}),
        (1, {22}),
    ]
    backend.abort_request("req-0")
    cpu = engine.storage_manager.local_cpu_backend
    for key in list(cpu.hot_cache):
        cpu.remove(key, force=True)
    req = replace(req, request_id="external", compute_start=end - 1, restore_end=end)
    _step(engine, backend, [req])
    assert [len(keys) for keys in store.gets] == [16, 1, 16, 1]
    _assert_remote(engine, req, store)


@pytest.mark.parametrize("failure", ["missing_row", "manifest", "load", "publication"])
def test_page_failure_does_not_log_commit_and_cleans_pending_sources(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    engine = page_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    if failure in ("load", "publication"):
        _step(engine, backend, [_request(request_id="seed")])
        backend.abort_request("seed")
    store = page_runtime.store
    store.puts.clear()
    logged = []
    monkeypatch.setattr(
        "lmcache_ascend.v1.layerwise_prefill_sync.logger.info",
        lambda message, *args: logged.append(message % args),
    )
    requests = [
        _request(start=300 if failure in ("load", "publication") else 0, end=530)
    ]
    if failure == "manifest":
        finish = backend.finish_step
        monkeypatch.setattr(backend, "finish_step", lambda: None)
        _step(engine, backend, requests)
        prefix = backend._prefixes["req-0", 1, 1, 21]
        prefix.keys[-1] = prefix.keys[-1].get_layer(0)
        with pytest.raises(ValueError, match="manifest"):
            finish()
    else:
        backend.bind_step(requests, _registry(engine))
        if failure == "missing_row":
            with pytest.raises(ValueError, match="Incomplete"):
                backend.finish_step()
        else:
            if failure == "load":
                engine.gpu_connector.fail_load = True
            else:
                monkeypatch.setattr(
                    engine,
                    "_make_shared_handles_for_layer",
                    Mock(side_effect=ValueError("publication failed")),
                )
            with pytest.raises(ValueError, match="sentinel|publication"):
                backend.wait_for_load(
                    _metadata(_view(), _view().rows_by_group[0][0], requests)
                )
    assert not store.puts
    assert backend._step_future.exception() is not None
    assert not backend._pending_rows
    assert not any("event=end" in line for line in logged)
    backend.abort_request("req-0")


@pytest.mark.parametrize("failure", [False, True])
def test_page_adapter_delayed_remote_never_records_early_commit(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    adapter = _make_adapter(page_runtime, monkeypatch)
    store = page_runtime.store
    req = _request()
    _save_rows(adapter, [req], _start(adapter, [req], final=True))
    record = Mock(wraps=adapter._record_prefill_save_group_completed)
    commit = Mock(wraps=adapter._mark_prefill_committed)
    monkeypatch.setattr(adapter, "_record_prefill_save_group_completed", record)
    monkeypatch.setattr(adapter, "_mark_prefill_committed", commit)
    store.gate.clear()
    store.fail = failure
    with ThreadPoolExecutor(max_workers=1) as pool:
        done = pool.submit(adapter.wait_for_save)
        assert store.submitted.wait(5)
        assert not done.done()
        assert not adapter.layerwise_prefill_request_persist_done("req-0")
        assert adapter.get_completed_decode_window_saves() == {}
        record.assert_not_called()
        commit.assert_not_called()
        store.gate.set()
        if failure:
            with pytest.raises(ValueError, match="page put failed"):
                done.result(20)
        else:
            done.result(20)
    if failure:
        record.assert_not_called()
        commit.assert_not_called()
        assert adapter.get_completed_decode_window_saves() == {}
        assert not adapter.layerwise_prefill_request_persist_done("req-0")
    else:
        assert record.call_count == 2
        commit.assert_called_once()


def test_bootstrap_unknown_fence_quarantines_unpublished_group(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = page_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    requests = [_request(start=300, end=530)]
    backend.bind_step(requests, _registry(engine))
    engine.gpu_connector.fail_load = True
    monkeypatch.setattr(
        engine.gpu_connector,
        "synchronize_dense_load_stream",
        Mock(side_effect=RuntimeError("unknown fence")),
    )
    with pytest.raises(LayerwisePrefillFenceError):
        backend.wait_for_load(_metadata(_view(), _view().rows_by_group[0][0], requests))
    assert not backend._pending_rows and not backend._prefixes
    assert len(backend._quarantined) == 79 * 2
    assert isinstance(backend._step_future.exception(), LayerwisePrefillFenceError)
    with pytest.raises(LayerwisePrefillFenceError, match="restart"):
        backend.abort_request("req-0")
    assert all(
        obj.is_valid() and obj.metadata.pin_count == 1 for obj in backend._quarantined
    )
    # Only the native-free test can establish safety after the simulated fence
    # failure. Production deliberately retains these owners until worker restart.
    backend._release(backend._quarantined)
    backend._quarantined.clear()
    backend._unsafe_transfer = False
    backend.abort_request("req-0")


@pytest.mark.parametrize("history_chunks", [1, 17])
def test_warm_commit_validation_is_bounded_by_changed_chunks(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, history_chunks: int
) -> None:
    engine = page_runtime.engine(blocks=80, slab_bytes=16 << 20)
    backend = engine.layerwise_prefill_window_backend
    first_end = history_chunks * 256 + 44
    next_end = first_end + 230
    req = replace(
        _request(),
        token_ids=tuple(range(next_end)),
        compute_end=first_end,
        block_ids_by_bank=tuple(
            (tuple(range(1 + bank * 38, 39 + bank * 38)),) * 2 for bank in (0, 1)
        ),
    )
    _step(engine, backend, [req])
    req = replace(
        req,
        compute_start=first_end,
        restore_end=first_end,
        compute_end=next_end,
        allocation_generation=2,
    )
    finish = backend.finish_step
    monkeypatch.setattr(backend, "finish_step", lambda: None)
    _step(engine, backend, [req])
    unchanged = {
        id(obj)
        for prefix in backend._prefixes.values()
        for obj in prefix.objects[:history_chunks]
    }
    object_type = type(next(iter(backend._prefixes.values())).objects[0])
    get_shape = object_type.get_shape
    checked = Counter()

    def check_shape(obj: Any) -> torch.Size:
        assert id(obj) not in unchanged, "Warm commit revisited unchanged objects"
        checked[id(obj)] += 1
        return get_shape(obj)

    with monkeypatch.context() as validation:
        validation.setattr(object_type, "get_shape", check_shape)
        validation.setattr(
            backend, "_plan", Mock(side_effect=AssertionError("Full row plan rebuild"))
        )
        metadata = Mock(wraps=engine.layerwise_prefill_row_metadata)
        positions = Mock(wraps=torch.arange)
        validation.setattr(engine, "layerwise_prefill_row_metadata", metadata)
        validation.setattr(torch, "arange", positions)
        finish()
        assert metadata.call_count == positions.call_count == 4
    assert len(checked) == 101 * 2
    assert set(checked.values()) == {1}
    assert backend._step_future.result() is None
    _assert_remote(engine, req, page_runtime.store)


@pytest.mark.parametrize(
    "mutation", ["count", "extent", "revision", "slab", "start", "end", "key"]
)
def test_warm_commit_rejects_invalid_manifest_before_remote_put(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    engine = page_runtime.engine()
    backend, store = engine.layerwise_prefill_window_backend, page_runtime.store
    _step(engine, backend, [_request()])
    finish = backend.finish_step
    monkeypatch.setattr(backend, "finish_step", lambda: None)
    _step(engine, backend, [_request(start=300, end=530, generation=2)])
    prefix = backend._prefixes["req-0", 2, 1, 21]
    if mutation == "count":
        prefix.starts.pop()
    elif mutation == "extent":
        prefix.ends[-1] -= 1
    elif mutation == "revision":
        prefix.revision -= 1
    elif mutation == "slab":
        prefix.slab_generation += 1
    elif mutation == "start":
        prefix.starts[1] += 1
    elif mutation == "end":
        prefix.ends[1] -= 1
    else:
        prefix.keys[1] = prefix.keys[1].get_layer(0)
    store.puts.clear()
    with pytest.raises(ValueError, match="Invalid page commit manifest"):
        finish()
    assert not store.puts
    assert backend._step_future.exception() is not None


def test_external_bootstrap_validates_unchanged_prefix_before_publication(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = page_runtime.engine()
    backend, store = engine.layerwise_prefill_window_backend, page_runtime.store
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    finish = backend.finish_step
    monkeypatch.setattr(backend, "finish_step", lambda: None)
    _step(engine, backend, [_request(request_id="external", start=300, end=530)])
    # The bootstrap's first chunk was restored, not recomputed. Unlike a warm
    # continuation it is untrusted and must be checked at the final barrier.
    prefix = backend._prefixes["external", 1, 1, 21]
    prefix.objects[0].metadata.cached_positions = torch.arange(1, 257)
    store.puts.clear()
    with pytest.raises(ValueError, match="Invalid page commit source metadata"):
        finish()
    assert not store.puts
    assert backend._step_future.exception() is not None


@pytest.mark.parametrize("missing_page", [False, True])
def test_real_gloo_page_only_bootstrap_and_warm_roundtrip(
    page_runtime: Any, tmp_path: Any, missing_page: bool
) -> None:
    # Page-byte comparisons exceed the CPU parallelization threshold. Do not
    # fork an active PyTorch worker pool and inherit its thread synchronization.
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = page_runtime.engine(size=2)
    root.storage_manager.local_cpu_backend.memory_allocator.buffer.share_memory_()
    passive = page_runtime.engine(1, root, size=2)
    context = multiprocessing.get_context("fork")
    results = context.Queue()

    def worker(rank: int, engine: Any) -> None:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            # Never submit work to an inherited event loop whose thread vanished
            # at fork. The parent constructed the connector but issued no I/O.
            if rank == 0:
                engine.storage_manager.storage_backends["RemoteBackend"].loop = loop
            torch.distributed.all_gather_object = page_runtime.real_gather
            torch.distributed.is_initialized = page_runtime.real_initialized
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{tmp_path}/page-gloo",
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
            seed = _request(request_id="seed")
            _step(engine, backend, [seed])
            backend.abort_request("seed")
            engine.layerwise_prefill_ack("seed retired")
            store = page_runtime.store
            if rank == 0:
                _assert_remote(engine, seed, store)
                cpu = engine.storage_manager.local_cpu_backend
                for key in list(cpu.hot_cache):
                    cpu.remove(key, force=True)
                assert cpu.memory_allocator.num_active_allocations == 0
                if missing_page:
                    del store.data[
                        next(
                            key
                            for key in store.data
                            if key.startswith("__lmcache_page_v1__")
                        )
                    ]
                store.puts.clear()
            engine.token_database = ChunkedTokenDatabase(engine.config, engine.metadata)
            req = _request(start=300, end=530)
            if missing_page:
                with pytest.raises(ValueError, match="Missing required"):
                    _step(engine, backend, [req])
                assert backend._step_future.exception() is not None
                assert not backend._pending_rows
                assert not store.puts
            else:
                _step(engine, backend, [req])
                assert backend._step_future.result() is None
                if rank == 0:
                    _assert_remote(engine, req, store)
                    assert len(store.gets) == 4  # A full page and tail per group.
                    store.puts.clear()
                req = _request(start=530, end=560, generation=2)
                _step(engine, backend, [req])
                assert backend._step_future.result() is None
                if rank == 0:
                    _assert_remote(engine, req, store)
                    assert len(store.gets) == 4
                    assert [kind for kind, _, _ in store.puts] == ["tail", "tail"]
            backend.abort_request("req-0")
            engine.layerwise_prefill_ack("request retired")
            if rank == 0:
                for key in list(cpu.hot_cache):
                    cpu.remove(key, force=True)
                assert cpu.memory_allocator.num_active_allocations == 0
            else:
                assert all(
                    not direction for direction, *_ in engine.gpu_connector.calls
                )
            results.put((rank, "closed" if missing_page else "roundtrip"))
            torch.distributed.destroy_process_group()
        except BaseException as exc:
            results.put((rank, repr(exc)))
            raise
        finally:
            asyncio.run_coroutine_threadsafe(
                loop.shutdown_default_executor(), loop
            ).result(10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

    processes = [
        context.Process(target=worker, args=(rank, engine))
        for rank, engine in enumerate((root, passive))
    ]
    for process in processes:
        process.start()
    try:
        outcomes = sorted(results.get(timeout=60) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert outcomes == [
            (rank, "closed" if missing_page else "roundtrip") for rank in (0, 1)
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
        torch.set_num_threads(num_threads)


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("mismatch", ["width", "cardinality", "format", "dtype"])
def test_page_bind_rejects_raw_abi_before_transfer(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, group: int, mismatch: str
) -> None:
    dims = {0: 2, 1: 1}
    if mismatch == "width":
        dims[group] += 1
    engine = page_runtime.engine(raw_dims=dims)
    backend = engine.layerwise_prefill_window_backend
    connection = engine.storage_manager.storage_backends["RemoteBackend"].connection
    native = connection.getWrappedConnector()
    if mismatch == "cardinality":
        engine.metadata.runtime_kv_group_layer_counts = (
            78 if group == 0 else 79,
            21 if group == 1 else 22,
        )
    elif mismatch == "dtype":
        native.meta_dtypes = [torch.float16]
    elif mismatch == "format":
        row_metadata = engine.layerwise_prefill_row_metadata

        def wrong_format(kv_group: int, tokens: int) -> tuple:
            shape, dtype, fmt = row_metadata(kv_group, tokens)
            if kv_group == group:
                fmt = (
                    MemoryFormat.KV_MLA_LATENT_FMT
                    if group == 1
                    else MemoryFormat.KV_DSA_INDEX_FMT
                )
            return shape, dtype, fmt

        monkeypatch.setattr(engine, "layerwise_prefill_row_metadata", wrong_format)
    with pytest.raises(ValueError, match="Page-first layout mismatch"):
        backend.bind_step([_request()], _registry(engine))
    assert engine.gpu_connector.layouts == {0: 79, 1: 22}
    assert not engine.gpu_connector.calls
    assert not page_runtime.store.puts and not page_runtime.store.gets
    assert not engine.storage_manager.local_cpu_backend.hot_cache


def test_page_raw_width_schema_tag_does_not_hide_layout_mismatch(
    page_runtime: Any,
) -> None:
    valid = page_runtime.engine()
    invalid = page_runtime.engine(raw_dims={0: 3, 1: 1})
    keys = [
        next(engine.token_database.process_tokens(list(range(256))))[2]
        for engine in (valid, invalid)
    ]
    assert dict(keys[0].tags)["payload_v2"] != dict(keys[1].tags)["payload_v2"]
    assert keys[0].chunk_hash == keys[1].chunk_hash
    assert mooncake_page_key(keys[0], 79) != mooncake_page_key(keys[1], 79)
    backend = invalid.layerwise_prefill_window_backend
    with pytest.raises(ValueError, match="Page-first layout mismatch"):
        backend.bind_step([_request()], _registry(invalid))
    _step(valid, valid.layerwise_prefill_window_backend, [_request(end=256)])
    assert [(kind, counts) for kind, _, counts in page_runtime.store.puts] == [
        ("page", (79,)),
        ("page", (22,)),
    ]
    _assert_remote(valid, _request(end=256), page_runtime.store)


def test_page_layout_failure_is_acknowledged_on_all_tp_ranks(page_runtime: Any) -> None:
    root = page_runtime.engine(size=2, raw_dims={0: 2, 1: 2})
    passive = page_runtime.engine(1, root, size=2, raw_dims={0: 2, 1: 2})
    page_runtime.thread.rank = 0
    rb = root.layerwise_prefill_window_backend
    page_runtime.thread.rank = 1
    pb = passive.layerwise_prefill_window_backend

    def bind(engine: Any, backend: Any) -> None:
        with pytest.raises(ValueError, match="Page-first layout mismatch"):
            backend.bind_step([_request()], _registry(engine))

    page_runtime.parallel(lambda: bind(root, rb), lambda: bind(passive, pb))
    assert not root.gpu_connector.calls and not passive.gpu_connector.calls
    assert not page_runtime.store.puts and not page_runtime.store.gets


def test_page_abi_matches_initialized_production_npu_row_shapes(
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = page_runtime.engine()
    registry = _registry(engine)
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.use_mla = connector.dsa_two_groups = True
    connector.use_gpu = False
    connector.lmcache_chunk_size = 256
    connector._group_layouts = {}
    connector._dsa_kv_topology_view = _view()
    engine.gpu_connector = connector
    sentinel = _patch_cpu_slot_preparer(monkeypatch, connector, registry)
    engine.layerwise_prefill_window_backend.bind_step([_request()], registry)
    assert len(sentinel.prepared) == 4
    assert [connector.get_num_layers(group) for group in (0, 1)] == [79, 22]
    assert [connector.get_shape(256, kv_group=group) for group in (0, 1)] == [
        torch.Size([512]),
        torch.Size([256]),
    ]
    assert not page_runtime.store.puts and not page_runtime.store.gets


@pytest.mark.parametrize("bad_rank", [0, 1])
@pytest.mark.parametrize(
    "phase,error",
    [
        ("load", "metadata"),
        ("load", "released"),
        ("save", "metadata"),
        ("save", "released"),
        ("save", "cursor"),
    ],
)
def test_adapter_coordinator_error_uses_existing_tp_validation_ack(
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    bad_rank: int,
    phase: str,
    error: str,
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    req = _request()
    callbacks = page_runtime.parallel(
        lambda: _start(adapters[0], [req]),
        lambda: _start(adapters[1], [req]),
    )
    if phase == "save":
        page_runtime.parallel(
            lambda: adapters[0].wait_for_layerwise_prefill_load(callbacks[0][0]),
            lambda: adapters[1].wait_for_layerwise_prefill_load(callbacks[1][0]),
        )
    metadata = [rows[0] for rows in callbacks]
    window = adapters[bad_rank]._layerwise_prefill_window
    if error == "metadata":
        metadata[bad_rank] = SimpleNamespace()
        message = "has no attribute 'row'"
    elif error == "released":
        # Diverge only coordinator state. Both backend manifests and callback
        # identities remain valid, so the rejection must be forwarded.
        window._released_generations[req.request_id] = req.allocation_generation
        message = "released generation"
    else:
        window._arenas[req.request_id].save_cursors[0] = 1
        message = "per-group row order"
    acknowledgements, forwarded = [], []
    for adapter in adapters:
        engine = adapter.lmcache_engine
        ack = Mock(wraps=engine.layerwise_prefill_ack)
        monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
        acknowledgements.append(ack)
        backend = adapter._layerwise_prefill_backend
        method = "wait_for_load" if phase == "load" else "sync_save"
        callback = Mock(wraps=getattr(backend, method))
        monkeypatch.setattr(backend, method, callback)
        forwarded.append(callback)
        monkeypatch.setattr(
            engine,
            "resolve_layerwise_prefill_row",
            Mock(side_effect=AssertionError("Validation failure published handles")),
        )

    def invoke(rank: int) -> None:
        adapter = adapters[rank]
        with pytest.raises(ValueError, match=message):
            if phase == "load":
                adapter.wait_for_layerwise_prefill_load(metadata[rank])
            else:
                planes = adapter.kv_caches[callbacks[rank][0].row.layer_name]
                adapter.save_layerwise_prefill_kv_layer(metadata[rank], planes)

    page_runtime.parallel(lambda: invoke(0), lambda: invoke(1))
    for rank, adapter in enumerate(adapters):
        acknowledgements[rank].assert_called_once()
        assert acknowledgements[rank].call_args.args[0][1][:2] == ("validate", phase)
        forwarded[rank].assert_called_once()
        validation = forwarded[rank].call_args.kwargs["validation_error"]
        assert (validation is not None) == (rank == bad_rank)
        backend = adapter._layerwise_prefill_backend
        assert backend._step_future.exception() is not None
        assert not backend._pending_rows
        assert not adapter.lmcache_engine.gpu_connector.calls
        assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
    assert not page_runtime.store.puts and not page_runtime.store.gets


@pytest.mark.parametrize("runtime_name", ["runtime", "page_runtime"])
@pytest.mark.parametrize("fail_load", [False, True])
def test_adapter_same_generation_bind_clears_previous_persist_done(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    runtime_name: str,
    fail_load: bool,
) -> None:
    runtime = request.getfixturevalue(runtime_name)
    adapter = _make_adapter(runtime, monkeypatch)
    req = _request()
    _save_rows(adapter, [req], _start(adapter, [req]))
    adapter.wait_for_save()
    assert adapter.layerwise_prefill_request_persist_done(req.request_id)
    req = replace(req, compute_start=300, restore_end=300, compute_end=530)
    callbacks = _start(adapter, [req], final=True)
    assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
    assert adapter.get_completed_decode_window_saves() == {}
    record = Mock(wraps=adapter._mark_prefill_committed)
    monkeypatch.setattr(adapter, "_mark_prefill_committed", record)
    if fail_load:
        adapter.lmcache_engine.gpu_connector.fail_load = True
        with pytest.raises(ValueError, match="sentinel failure"):
            adapter.wait_for_layerwise_prefill_load(callbacks[0])
        assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
        with pytest.raises(ValueError, match="Incomplete"):
            adapter.wait_for_save()
        record.assert_not_called()
        assert adapter.get_completed_decode_window_saves() == {}
        assert not adapter._layerwise_prefill_window.has_request(req.request_id)
        assert not adapter._layerwise_prefill_backend._prefixes
    else:
        _save_rows(adapter, [req], callbacks)
        adapter.wait_for_save()
        record.assert_called_once()
        assert adapter.layerwise_prefill_request_persist_done(req.request_id)
        assert adapter.get_completed_decode_window_saves() == {req.request_id: 512}


@pytest.mark.parametrize("saved_rows", [1, 101])
def test_adapter_rejects_unfinished_previous_bind_on_all_ranks(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, saved_rows: int
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    req = _request()
    page_runtime.parallel(
        lambda: _save_rows(adapters[0], [req], _start(adapters[0], [req])[:saved_rows]),
        lambda: _save_rows(adapters[1], [req], _start(adapters[1], [req])[:saved_rows]),
    )
    acknowledgements = []
    calls = [list(adapter.lmcache_engine.gpu_connector.calls) for adapter in adapters]
    for adapter in adapters:
        ack = Mock(wraps=adapter.lmcache_engine.layerwise_prefill_ack)
        monkeypatch.setattr(adapter.lmcache_engine, "layerwise_prefill_ack", ack)
        acknowledgements.append(ack)
    next_req = replace(req, compute_start=300, restore_end=300, compute_end=530)

    def bind(rank: int) -> None:
        with pytest.raises(
            ValueError, match="Previous synchronous step is not persisted"
        ):
            _start(adapters[rank], [next_req])

    page_runtime.parallel(lambda: bind(0), lambda: bind(1))
    for rank, adapter in enumerate(adapters):
        acknowledgements[rank].assert_called_once()
        assert acknowledgements[rank].call_args.args[0][0] == "bind"
        assert adapter.lmcache_engine.gpu_connector.calls == calls[rank]
        assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
        assert adapter.get_completed_decode_window_saves() == {}
        adapter._layerwise_prefill_window.release_request(req.request_id)
    assert not page_runtime.store.puts


def test_adapter_incomplete_page_step_releases_pending_group_without_barrier(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _make_adapter(page_runtime, monkeypatch)
    engine, backend = adapter.lmcache_engine, adapter._layerwise_prefill_backend
    _step(engine, backend, [_request(request_id="seed")])
    backend.abort_request("seed")
    req = _request(start=300, end=530)
    callbacks = _start(adapter, [req])
    adapter.wait_for_layerwise_prefill_load(callbacks[0])
    assert len(backend._pending_rows) == 78
    owners = [obj for row in backend._pending_rows.values() for obj in row]
    owners.extend(
        obj for prefix in backend._prefixes.values() for obj in prefix.objects
    )
    barrier = Mock(side_effect=AssertionError("Failure cleanup waited for success"))
    window = adapter._layerwise_prefill_window
    monkeypatch.setattr(window, "wait_for_request_persist_done", barrier)
    with pytest.raises(ValueError, match="Incomplete"):
        adapter.wait_for_save()
    barrier.assert_not_called()
    assert not window.has_request(req.request_id)
    assert not backend._prefixes and not backend._pending_rows
    assert all(
        obj.metadata.pin_count == 0 and obj.get_ref_count() == 1 for obj in owners
    )
    assert backend._step_future.exception() is not None


def test_adapter_fatal_page_fence_retains_ownership_on_all_ranks(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    backends = [adapter._layerwise_prefill_backend for adapter in adapters]
    engines = [adapter.lmcache_engine for adapter in adapters]
    page_runtime.parallel(
        lambda: _step(engines[0], backends[0], [_request(request_id="seed")]),
        lambda: _step(engines[1], backends[1], [_request(request_id="seed")]),
    )
    for backend in backends:
        backend.abort_request("seed")
    req = _request(start=300, end=530)
    callbacks = page_runtime.parallel(
        lambda: _start(adapters[0], [req]), lambda: _start(adapters[1], [req])
    )
    page_runtime.parallel(
        lambda: _save_rows(adapters[0], [req], callbacks[0][:1]),
        lambda: _save_rows(adapters[1], [req], callbacks[1][:1]),
    )
    engines[0].gpu_connector.fail_load = True
    monkeypatch.setattr(
        engines[0].gpu_connector,
        "synchronize_dense_load_stream",
        Mock(side_effect=RuntimeError("unknown fence")),
    )

    def load(rank: int) -> None:
        with pytest.raises(LayerwisePrefillFenceError):
            adapters[rank].wait_for_layerwise_prefill_load(callbacks[rank][1])

    def finish(rank: int) -> None:
        with pytest.raises(LayerwisePrefillFenceError, match="restart"):
            adapters[rank].wait_for_save()

    try:
        page_runtime.parallel(lambda: load(0), lambda: load(1))
        owners = [
            backend._quarantined
            + [
                obj for prefix in backend._prefixes.values() for obj in prefix.objects
            ]
            for backend in backends
        ]
        assert all(backend._quarantined for backend in backends)
        page_runtime.parallel(lambda: finish(0), lambda: finish(1))
        for adapter, backend, objects in zip(adapters, backends, owners, strict=True):
            assert adapter._layerwise_prefill_window.has_request(req.request_id)
            assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
            assert backend._quarantined and backend._prefixes
            assert all(
                obj.is_valid() and obj.metadata.pin_count == 1 for obj in objects
            )
            with pytest.raises(LayerwisePrefillFenceError, match="restart"):
                adapter._layerwise_prefill_window.release_request(req.request_id)
    finally:
        # Native-free CPU transfers have finished; only the test may recover
        # from its simulated unknown fence instead of restarting the worker.
        for adapter, backend in zip(adapters, backends, strict=True):
            backend._release(backend._quarantined)
            backend._quarantined.clear()
            backend._unsafe_transfer = False
            adapter._layerwise_prefill_window.release_request(req.request_id)


def test_real_gloo_adapter_forwards_coordinator_save_cursor_error(
    page_runtime: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    context = multiprocessing.get_context("fork")
    results = context.Queue()

    def worker(rank: int, adapter: Any) -> None:
        try:
            torch.distributed.all_gather_object = page_runtime.real_gather
            torch.distributed.is_initialized = page_runtime.real_initialized
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{tmp_path}/coordinator-gloo",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=10),
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

            engine = adapter.lmcache_engine
            engine.broadcast_object_fn = broadcast
            metadata = _start(adapter, [_request()])[0]
            # Empty initial history needs no native I/O or shared row views.
            # Only the real CPU control channels are used before rejection.
            adapter.wait_for_layerwise_prefill_load(metadata)
            if rank == 1:
                adapter._layerwise_prefill_window._arenas["req-0"].save_cursors[0] = 1
            ack = Mock(wraps=engine.layerwise_prefill_ack)
            engine.layerwise_prefill_ack = ack
            with pytest.raises(ValueError, match="per-group row order"):
                adapter.save_layerwise_prefill_kv_layer(
                    metadata, adapter.kv_caches[metadata.row.layer_name]
                )
            ack.assert_called_once()
            assert ack.call_args.args[0][1][:2] == ("validate", "save")
            assert not engine.gpu_connector.calls
            assert not page_runtime.store.puts and not page_runtime.store.gets
            assert not adapter.layerwise_prefill_request_persist_done("req-0")
            adapter._layerwise_prefill_window.release_request("req-0")
            results.put((rank, "rejected"))
            torch.distributed.destroy_process_group()
        except BaseException as exc:
            results.put((rank, repr(exc)))
            raise

    processes = [
        context.Process(target=worker, args=(rank, adapter))
        for rank, adapter in enumerate(adapters)
    ]
    for process in processes:
        process.start()
    try:
        outcomes = sorted(results.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert outcomes == [(0, "rejected"), (1, "rejected")]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


@pytest.mark.parametrize("bad_rank", [0, 1])
@pytest.mark.parametrize(
    "failure,message",
    [
        ("callbacks", "no canonical row callbacks"),
        ("bindings", "lost its bank-aware request bindings"),
        ("null_binding", "request_id"),
        ("unhashable_binding", "unhashable type"),
        ("attention", "requires a model forward"),
        ("duplicate_ids", "identities differ from scheduler bindings"),
        ("identity", "identities differ from scheduler bindings"),
        ("generation", "identities differ from scheduler bindings"),
        ("resume", "Cannot resume an unfinished P step"),
    ],
)
def test_adapter_startup_validation_reaches_both_ranks_before_transfer(
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    bad_rank: int,
    failure: str,
    message: str,
) -> None:
    adapters = _make_tp_adapters(page_runtime, monkeypatch)
    req = _request()
    result_guards = []
    if failure == "resume":
        page_runtime.parallel(
            lambda: _save_rows(adapters[0], [req], _start(adapters[0], [req])[:1]),
            lambda: _save_rows(adapters[1], [req], _start(adapters[1], [req])[:1]),
        )
        for adapter in adapters:
            future = adapter._layerwise_prefill_backend._step_future
            assert not future.done()
            guard = Mock(
                side_effect=AssertionError("Resume waited for unfinished persistence")
            )
            monkeypatch.setattr(future, "result", guard)
            result_guards.append(guard)
    before = [list(adapter.lmcache_engine.gpu_connector.calls) for adapter in adapters]
    tracked = []
    for adapter in adapters:
        window = adapter._layerwise_prefill_window
        spies = []
        for owner, name in (
            (window, "bind_step"),
            (adapter._layerwise_prefill_backend, "bind_step"),
            (adapter.lmcache_engine, "layerwise_prefill_ack"),
            (adapter, "_mark_prefill_committed"),
            (window, "wait_for_request_persist_done"),
        ):
            spy = Mock(wraps=getattr(owner, name))
            monkeypatch.setattr(owner, name, spy)
            spies.append(spy)
        tracked.append(spies)
    bad = adapters[bad_rank]
    start = bad.start_load_kv

    def invalid_start(forward: Any) -> None:
        metadata = bad._parent._connector_metadata
        if failure == "callbacks":
            forward.attn_metadata = {"unrelated": SimpleNamespace()}
        elif failure == "bindings":
            metadata.layerwise_prefill_requests = None
        elif failure == "null_binding":
            metadata.layerwise_prefill_requests = [None]
        elif failure == "unhashable_binding":
            metadata.layerwise_prefill_requests = [
                SimpleNamespace(request_id=[], allocation_generation=1)
            ]
        elif failure == "attention":
            forward.attn_metadata = None
        elif failure == "duplicate_ids":
            metadata.layerwise_prefill_requests.append(req)
        elif failure == "resume":
            metadata.requests[0].resumed_from_preemption = True
        else:
            callbacks = next(
                item.layerwise_prefill_callback_metadata
                for item in forward.attn_metadata.values()
                if getattr(item, "layerwise_prefill_callback_metadata", ())
            )
            callbacks[0].request_generations = (
                ("other", req.allocation_generation)
                if failure == "identity"
                else (req.request_id, req.allocation_generation + 1),
            )
        start(forward)

    monkeypatch.setattr(bad, "start_load_kv", invalid_start)

    def invoke(rank: int) -> None:
        with pytest.raises(ValueError, match=message):
            _start(adapters[rank], [req], final=True)

    page_runtime.parallel(lambda: invoke(0), lambda: invoke(1))
    for rank, adapter in enumerate(adapters):
        coordinator_bind, backend_bind, ack, commit, barrier = tracked[rank]
        coordinator_bind.assert_called_once()
        backend_bind.assert_called_once()
        ack.assert_called_once()
        assert ack.call_args.args[0][0] == "bind"
        validation = coordinator_bind.call_args.kwargs["validation_error"]
        assert (validation is not None) == (rank == bad_rank)
        if rank == bad_rank:
            assert message in str(validation)
            assert backend_bind.call_args.kwargs["validation_error"] is validation
        elif failure != "resume":
            assert backend_bind.call_args.kwargs["validation_error"] is None
        commit.assert_not_called()
        barrier.assert_not_called()
        assert adapter.lmcache_engine.gpu_connector.calls == before[rank]
        assert not adapter.layerwise_prefill_request_persist_done(req.request_id)
        assert adapter.get_completed_decode_window_saves() == {}
        assert not adapter._prefill_save_completed_groups
    for guard in result_guards:
        guard.assert_not_called()
    assert not page_runtime.store.puts and not page_runtime.store.gets
    if failure == "resume":
        for adapter in adapters:
            adapter._layerwise_prefill_window.release_request(req.request_id)
