# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401

# TODO (gingfung): once we supported NPUDirectFS,
# re-enable test_multi_device_backends

# Standard
from types import SimpleNamespace

# Third Party
from lmcache_tests.v1.test_cache_engine import (
    test_builder,
    test_builder_destroy,
    test_builder_destroy_multiple_instances,
    test_force_store_wait,
    test_paged_hierarchy_retrieve,
    test_paged_mem_leak,
    test_paged_mixed_retrieve,
    test_paged_prefetch_retrieve,
    test_paged_retrieve_after_eviction,
)
from lmcache_tests.v1.test_cache_engine import (
    test_paged_retrieve_prefix as original_paged_retrieve_prefix,
)
from lmcache_tests.v1.test_cache_engine import (
    test_paged_same_retrieve_store,
    test_paged_store_kv_tensors_mask,
    test_paged_store_offset,
)
import pytest
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cpu", "local_disk", "remote"])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_retrieve_prefix_patched(
    chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
):
    original_paged_retrieve_prefix(
        chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
    )


def test_prepare_dsa_store_exchange_initializes_group_layout() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    calls = []
    engine.gpu_connector = SimpleNamespace(
        _group_layouts={1: object()},
        kv_device=torch.device("meta"),
        append_sparse_chunk_ptr_cache_for_layer=lambda *args: None,
    )
    engine._ensure_layerwise_connector_layout = lambda **kwargs: calls.append(
        kwargs
    )
    kvcaches = [torch.zeros(1)]

    engine.prepare_dsa_store_exchange(kvcaches=kvcaches, kv_group=1)

    assert calls == [{"kvcaches": kvcaches, "kv_group": 1}]


def test_prepare_dsa_store_exchange_rejects_missing_npu_device() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.gpu_connector = SimpleNamespace(
        _group_layouts={0: object()},
        kv_device=None,
        append_sparse_chunk_ptr_cache_for_layer=lambda *args: None,
    )
    engine._ensure_layerwise_connector_layout = lambda **kwargs: None

    with pytest.raises(RuntimeError, match="NPU device is unavailable"):
        engine.prepare_dsa_store_exchange(
            kvcaches=[torch.zeros(1)],
            kv_group=0,
        )
