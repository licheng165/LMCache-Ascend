# SPDX-License-Identifier: Apache-2.0
"""Real NPU/native row roundtrip with post-SFA and cross-bank dependencies."""

# Standard
import ctypes

# Third Party
import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or torch.npu.is_available() is not True,
    reason="asynchronous native row transfer requires an Ascend NPU",
)


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("prepared_slots", [False, True])
@torch.inference_mode()
def test_native_async_row_roundtrip_with_tail_and_compute_consumer(
    group, prepared_slots
):
    from torch.utils._python_dispatch import TorchDispatchMode

    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )
    import lmcache_ascend.c_ops as lmc_ops

    class NoKVCopy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func._schema.name in ("aten::_to_copy", "aten::copy_", "aten::clone"):
                assert not any(
                    isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16
                    for t in args
                ), "KV staging/D2D copy inside a row hook"
            return func(*args, **(kwargs or {}))

    device = torch.device("npu", torch.npu.current_device())
    widths = (512, 64) if group == 0 else (128,)
    shape_prefix = (3, 16, 1)
    src = tuple(
        torch.full((*shape_prefix, width), -3, dtype=torch.bfloat16, device=device)
        for width in widths
    )
    dst = tuple(torch.full_like(t, -7) for t in src)
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._group_layouts = {}
    connector.use_mla, connector.use_gpu, connector.dsa_two_groups = True, False, True
    connector.dtype = torch.bfloat16
    connector.num_layers, connector.lmcache_chunk_size = 79, 4
    connector.store_stream = torch.npu.Stream(device=device)
    connector.load_stream = torch.npu.Stream(device=device)
    connector._lazy_initialize_buffer(
        [src] * (79 if group == 0 else 22),
        kv_group=group,
        init_staging=False,
    )

    # Native registered host allocation, not torch's ordinary pinned allocator.
    # Free only after drain succeeds, including assertion/submit error paths.
    ptr = lmc_ops.alloc_pinned_ptr(64 * 1024, 0)
    try:
        buffer = (ctypes.c_ubyte * (64 * 1024)).from_address(ptr)
        host = torch.frombuffer(buffer, dtype=torch.bfloat16)
        chunks, offset = [], 128
        for size in (4, 1):
            count = size * sum(widths)
            chunks.append(host[offset : offset + count])
            chunks[-1].fill_(-11)
            offset += count + 128
        slots = torch.tensor([47, 0, 19, 2, 35], dtype=torch.int64)
        mapping = (
            connector.prepare_layerwise_prefill_slots(
                slots,
                kv_group=group,
                capacity=48,
            )
            if prepared_slots
            else slots
        )

        with NoKVCopy():
            store = connector.prepare_layerwise_prefill_row(
                src,
                chunks,
                [100, 104],
                [104, 105],
                mapping,
                kv_group=group,
                direction=True,
                slot_mapping_base=100,
            )
            load = connector.prepare_layerwise_prefill_row(
                dst,
                chunks,
                [100, 104],
                [104, 105],
                mapping,
                kv_group=group,
                direction=False,
                slot_mapping_base=100,
            )
        # A producer queued after preparation must be covered by submit's wait.
        for index, plane in enumerate(src):
            plane.fill_(10 + index)
        with NoKVCopy():
            connector.submit_layerwise_prefill_row(store)
            connector.submit_layerwise_prefill_row(load, wait_event=store.done_event)
            connector.wait_layerwise_prefill_row(load)
        # Read on compute before any host fence. The result checks the wait,
        # rather than relying on a later complete() to mask a missing dependency.
        consumed = tuple(plane + 1 for plane in dst)
        connector.complete_layerwise_prefill_row(load)
        connector.complete_layerwise_prefill_row(store)
        for index, (actual, width) in enumerate(zip(consumed, widths, strict=True)):
            expected = torch.full((48, width), -6, dtype=torch.bfloat16)
            expected[slots] = 11 + index
            torch.testing.assert_close(
                actual.cpu().view(48, width), expected, rtol=0, atol=0
            )
        for chunk, size in zip(chunks, (4, 1), strict=True):
            offset = 0
            for index, width in enumerate(widths):
                assert bool((chunk[offset : offset + size * width] == 10 + index).all())
                offset += size * width
        # Completion releases payloads but not the event needed by later banks.
        with NoKVCopy():
            connector.wait_layerwise_prefill_row(store)
        assert not store._owners and not load._owners
    finally:
        connector.drain_layerwise_prefill_transfers()
        lmc_ops.free_pinned_ptr(ptr)
