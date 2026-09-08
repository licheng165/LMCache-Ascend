# SPDX-License-Identifier: Apache-2.0
"""CPU-backed NPU tensors with the real wrapper and a deferred native mock."""

# Standard
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import copy
import gc
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector import npu_connectors
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


_DEVICE = SimpleNamespace(type="npu", index=0)
_OTHER_DEVICE = SimpleNamespace(type="npu", index=1)


class _NPUTensor(torch.Tensor):
    @property
    def device(self):
        return _DEVICE


class _OtherNPUTensor(_NPUTensor):
    @property
    def device(self):
        return _OTHER_DEVICE


class _Stream:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.device = _DEVICE

    def wait_stream(self, producer):
        assert self.env.active is self
        self.env.events.append((self.name, "wait", producer.name))

    def synchronize(self):
        self.env.events.append((self.name, "sync"))
        if self.env.sync_error:
            raise RuntimeError("completion failed")
        for operation in self.env.pending:
            operation()
        self.env.pending.clear()


@pytest.fixture
def row_env(monkeypatch):
    env = SimpleNamespace(
        events=[],
        calls=[],
        pending=[],
        active=None,
        launch_error=False,
        sync_error=False,
        pointer_error=None,
        host_ops=dict(
            cat=0,
            readback=0,
            tolist=0,
            slices=0,
            uploads=0,
            validation_scans=0,
            unique_checks=0,
        ),
    )
    env.compute = _Stream(env, "compute")
    env.store = _Stream(env, "store")
    env.load = _Stream(env, "load")

    @contextmanager
    def stream_context(stream):
        assert env.active is None
        env.active = stream
        env.events.append((stream.name, "enter"))
        try:
            yield
        finally:
            env.events.append((stream.name, "exit"))
            env.active = None

    @contextmanager
    def device_context(device):
        assert device == _DEVICE
        yield

    def current_stream(device):
        assert device == _DEVICE
        assert env.active is None
        return env.compute

    monkeypatch.setattr(torch.npu, "stream", stream_context)
    monkeypatch.setattr(torch.npu, "device", device_context)
    monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    tensor_factory, tensor_to, tensor_cat = torch.tensor, torch.Tensor.to, torch.cat
    tensor_tolist, tensor_getitem = torch.Tensor.tolist, torch.Tensor.__getitem__

    def make_tensor(*args, **kwargs):
        if kwargs.get("device") is _DEVICE:
            assert env.active is not None
            assert kwargs["dtype"] in (torch.int32, torch.int64)
            env.events.append((env.active.name, "metadata", kwargs["dtype"]))
            kwargs["device"] = "cpu"
        return tensor_factory(*args, **kwargs)

    def to_tensor(tensor, *args, **kwargs):
        if kwargs.get("device") is _DEVICE:
            assert tensor.dtype in (torch.int32, torch.int64), "KV staging forbidden"
            env.events.append(((env.active or env.compute).name, "slots"))
            env.host_ops["uploads"] += 1
            kwargs["device"] = "cpu"
            tensor = tensor.as_subclass(torch.Tensor)
            return tensor_to(tensor, *args, **kwargs).as_subclass(_NPUTensor)
        if kwargs.get("device") == "cpu" and isinstance(tensor, _NPUTensor):
            env.host_ops["readback"] += 1
            tensor = tensor.as_subclass(torch.Tensor)
        return tensor_to(tensor, *args, **kwargs)

    def cat_tensors(tensors, *args, **kwargs):
        assert env.active is not None
        assert all(t.dtype in (torch.int32, torch.int64) for t in tensors)
        env.events.append((env.active.name, "pack_slots"))
        env.host_ops["cat"] += 1
        return tensor_cat(tensors, *args, **kwargs)

    class SlotValues(list):
        def __iter__(self):
            env.host_ops["validation_scans"] += 1
            return super().__iter__()

    def tolist(tensor):
        env.host_ops["tolist"] += 1
        return SlotValues(tensor_tolist(tensor))

    def getitem(tensor, index):
        if tensor.dtype in (torch.int32, torch.int64) and isinstance(index, slice):
            env.host_ops["slices"] += 1
        if isinstance(tensor, _NPUTensor):
            return tensor_getitem(tensor.as_subclass(torch.Tensor), index).as_subclass(
                type(tensor)
            )
        return tensor_getitem(tensor, index)

    def unique_slots(values=()):
        env.host_ops["unique_checks"] += 1
        return set(values)

    monkeypatch.setattr(torch, "tensor", make_tensor)
    monkeypatch.setattr(torch.Tensor, "to", to_tensor)
    monkeypatch.setattr(torch, "cat", cat_tensors)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    monkeypatch.setattr(npu_connectors, "set", unique_slots, raising=False)

    def get_device_ptr(ptr):
        assert env.active is not None
        env.events.append((env.active.name, "registered", ptr))
        if env.pointer_error == "raise":
            raise RuntimeError("registration lookup failed")
        if env.pointer_error == "null":
            return None
        if env.pointer_error is not None:
            return env.pointer_error
        return ptr + 4096

    monkeypatch.setattr(npu_connectors.lmc_ops, "get_device_ptr", get_device_ptr)

    def native(
        chunks,
        planes,
        slots,
        offsets,
        sizes,
        total,
        fmt,
        token_major,
        two_major,
        k_width,
        v_width,
        dsa_width,
        host_interleaved,
        direction,
        pointers,
        fixed_chunk_size,
    ):
        assert env.active is (env.store if direction else env.load)
        assert not token_major and not two_major and not host_interleaved
        assert fixed_chunk_size == 0
        assert offsets.dtype == sizes.dtype == torch.int32
        assert pointers.dtype == slots.dtype == torch.int64
        assert all(t.is_contiguous() for t in (slots, offsets, sizes, pointers))
        assert tensor_tolist(pointers) == [chunk.data_ptr() + 4096 for chunk in chunks]
        env.events.append((env.active.name, "launch"))
        env.calls.append(
            dict(
                slots=tensor_tolist(slots.as_subclass(torch.Tensor)),
                offsets=tensor_tolist(offsets),
                sizes=tensor_tolist(sizes),
                slot_ptr=slots.data_ptr(),
                total=total,
                fmt=fmt,
                widths=(k_width, v_width, dsa_width),
                direction=direction,
                plane_ptrs=[plane.data_ptr() for plane in planes],
                chunk_ptrs=[chunk.data_ptr() for chunk in chunks],
            )
        )
        # The native ABI only owns raw pointers. Do not let the mock hide a
        # premature Python-owner release by holding strong tensor references.
        plane_refs = [weakref.ref(t) for t in planes]
        chunk_refs = [weakref.ref(t) for t in chunks]
        metadata_refs = [weakref.ref(t) for t in (slots, offsets, sizes, pointers)]
        env.last_refs = plane_refs + chunk_refs + metadata_refs

        def complete():
            assert all(ref() is not None for ref in env.last_refs)
            slot_values = tensor_tolist(metadata_refs[0]().as_subclass(torch.Tensor))
            offset_values = tensor_tolist(metadata_refs[1]())
            size_values = tensor_tolist(metadata_refs[2]())
            for ref, offset, size in zip(
                chunk_refs, offset_values, size_values, strict=True
            ):
                chunk = ref().reshape(-1)
                plane_offset = 0
                for plane_ref in plane_refs:
                    plane = plane_ref().as_subclass(torch.Tensor)
                    plane = plane.view(plane.shape[0] * plane.shape[1], -1)
                    width = plane.shape[1]
                    packed = chunk[plane_offset : plane_offset + size * width]
                    packed = packed.view(size, width)
                    for token, slot in enumerate(slot_values[offset : offset + size]):
                        if direction:
                            packed[token].copy_(plane[slot])
                        else:
                            plane[slot].copy_(packed[token])
                    plane_offset += size * width

        env.pending.append(complete)
        if env.launch_error:
            raise RuntimeError("native launch failed")

    monkeypatch.setattr(
        npu_connectors.lmc_ops, "dense_mla_dsa_batched_direct_kv_transfer", native
    )

    def make_planes(group, blocks=3):
        shapes = (
            ((blocks, 4, 2, 32), (blocks, 4, 1, 16))
            if group == 0
            else ((blocks, 4, 1, 32),)
        )
        planes = []
        for index, shape in enumerate(shapes):
            count = torch.Size(shape).numel()
            backing = torch.arange(count + 32, dtype=torch.float32).to(torch.bfloat16)
            backing.add_(index * 2048)
            # All planes are views with a nonzero storage offset.
            planes.append(backing[16 : 16 + count].view(shape).as_subclass(_NPUTensor))
        return tuple(planes)

    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._group_layouts = {}
    connector.use_mla = True
    connector.use_gpu = False
    connector.dsa_two_groups = True
    connector.dtype = torch.bfloat16
    connector.num_layers = 79
    connector.lmcache_chunk_size = 4
    connector.store_stream, connector.load_stream = env.store, env.load
    for group, count in ((0, 79), (1, 22)):
        connector._lazy_initialize_buffer(
            [make_planes(group)] * count, kv_group=group, init_staging=False
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("A row hook must not initialize/iterate global caches or stage KV")

    connector.kvcaches = SimpleNamespace()
    connector.initialize_kvcaches_ptr = forbidden
    connector._lazy_initialize_buffer = forbidden
    connector._allocate_layerwise_staging_buffer = forbidden
    env.connector = connector
    env.planes = make_planes
    env.host_ops = dict.fromkeys(env.host_ops, 0)
    return env


def _chunks(planes, sizes):
    width = sum(plane.shape[-2] * plane.shape[-1] for plane in planes)
    return [
        torch.full((size * width + 8,), -7, dtype=torch.bfloat16)[8:] for size in sizes
    ]


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("direction", [True, False])
@pytest.mark.parametrize("sizes", [(3, 1), (1, 3, 2)])
def test_row_transfer_packed_tails_and_absolute_slot_metadata(
    row_env, group, direction, sizes
):
    env = row_env
    planes = env.planes(group)
    chunks = _chunks(planes, sizes)
    starts, ends, cursor = [], [], 100
    for size in sizes:
        starts.append(cursor)
        ends.append(cursor + size)
        cursor += size + 1
    slots = torch.tensor([10, 2, 7, 6, 8, 0, 4, 9, 3, 1], dtype=torch.int32)
    # Place sentinels only in gaps, which must not reach the native kernel.
    for end in ends:
        slots[end - 100] = -1
    expected_slots = [
        int(slots[i])
        for start, end in zip(starts, ends, strict=True)
        for i in range(start - 100, end - 100)
    ]
    snapshots = [p.as_subclass(torch.Tensor).clone() for p in planes]
    if not direction:
        for i, chunk in enumerate(chunks):
            chunk.copy_(torch.arange(chunk.numel(), dtype=torch.float32) + i * 1024)

    result = env.connector.transfer_layerwise_prefill_row(
        planes,
        chunks,
        starts,
        ends,
        slots,
        kv_group=group,
        direction=direction,
        slot_mapping_base=100,
    )

    assert result is None and not env.pending
    assert len(env.calls) == 1
    call = env.calls[0]
    assert call["slots"] == expected_slots
    assert call["sizes"] == list(sizes)
    assert call["offsets"] == [sum(sizes[:i]) for i in range(len(sizes))]
    assert call["total"] == sum(sizes)
    assert call["widths"] == ((64, 16, 0) if group == 0 else (32, 0, 32))
    assert (
        call["fmt"]
        == (KVCacheFormat.MLA_LATENT if group == 0 else KVCacheFormat.DSA_INDEX).value
    )
    assert call["plane_ptrs"] == [p.data_ptr() for p in planes]
    assert all(p.data_ptr() != p.untyped_storage().data_ptr() for p in planes)
    assert all(c.data_ptr() != c.untyped_storage().data_ptr() for c in chunks)
    stream_name = "store" if direction else "load"
    assert env.events[:2] == [(stream_name, "enter"), (stream_name, "wait", "compute")]
    assert env.events[-3:] == [
        (stream_name, "launch"),
        (stream_name, "exit"),
        (stream_name, "sync"),
    ]
    assert env.connector.get_num_layers(0) == 79
    assert env.connector.get_num_layers(1) == 22
    assert env.connector._current_kv_group == 1

    token_offset = 0
    for chunk, size in zip(chunks, sizes, strict=True):
        plane_offset = 0
        for plane, snapshot in zip(planes, snapshots, strict=True):
            flat = plane.as_subclass(torch.Tensor).view(12, -1)
            width = flat.shape[1]
            packed = chunk[plane_offset : plane_offset + size * width].view(size, width)
            selected = expected_slots[token_offset : token_offset + size]
            if direction:
                torch.testing.assert_close(packed, snapshot.view(12, -1)[selected])
                torch.testing.assert_close(plane.as_subclass(torch.Tensor), snapshot)
            else:
                torch.testing.assert_close(flat[selected], packed)
                untouched = sorted(set(range(12)) - set(expected_slots))
                torch.testing.assert_close(
                    flat[untouched], snapshot.view(12, -1)[untouched]
                )
            plane_offset += size * width
        token_offset += size


def test_npu_noncontiguous_slot_mapping_is_packed_on_transfer_stream(row_env):
    env = row_env
    planes = env.planes(1)
    mapping = torch.tensor([4, -1, 0, -1, 9, -1], dtype=torch.int32)
    mapping = mapping[::2].as_subclass(_NPUTensor)
    env.connector.transfer_layerwise_prefill_row(
        planes,
        _chunks(planes, [3]),
        [0],
        [3],
        mapping,
        kv_group=1,
        direction=False,
    )
    assert env.calls[0]["slots"] == [4, 0, 9]


@pytest.mark.parametrize(
    "slots, direction, error",
    [
        ([-1, 0], True, "capacity"),
        ([0, 12], False, "capacity"),
        ([2, 2], False, "duplicate"),
        ([2, 2], True, None),
    ],
)
def test_slot_bounds_and_h2d_duplicates(row_env, slots, direction, error):
    env = row_env
    planes = env.planes(0)
    args = (planes, _chunks(planes, [1, 1]), [0, 1], [1, 2], torch.tensor(slots))
    if error:
        with pytest.raises(ValueError, match=error):
            env.connector.transfer_layerwise_prefill_row(
                *args, kv_group=0, direction=direction
            )
        assert not env.calls
    else:
        env.connector.transfer_layerwise_prefill_row(
            *args, kv_group=0, direction=direction
        )
        assert len(env.calls) == 1
    assert env.events[-1] == ("store" if direction else "load", "sync")


@pytest.mark.parametrize(
    "case, error",
    [
        ("missing_group", "registered full group"),
        ("empty_layout", "registered full group"),
        ("layout_indices", "registered full group"),
        ("layout_dims", "positive PA-BSND"),
        ("format", "MLA/DSA"),
        ("two_major", "PA-BSND"),
        ("layout_device", "NPU layout"),
        ("connector_dtype", "BF16"),
        ("plane_count", "exactly"),
        ("plane_object", "KV planes"),
        ("plane_dtype", "KV planes"),
        ("plane_cpu", "KV planes"),
        ("plane_device", "KV planes"),
        ("plane_rank", "KV planes"),
        ("plane_stride", "KV planes"),
        ("plane_width", "KV planes"),
        ("plane_blocks", "KV planes"),
        ("plane_zero", "KV planes"),
        ("chunk_count", "counts"),
        ("chunk_short", "exact packed"),
        ("chunk_padded_tail", "exact packed"),
        ("chunk_stride", "contiguous BF16"),
        ("chunk_dtype", "contiguous BF16"),
        ("chunk_npu", "local contiguous"),
        ("chunk_object", "local contiguous"),
        ("base_negative", "slot_mapping_base"),
        ("base_float", "slot_mapping_base"),
        ("range_before_base", "positive ranges"),
        ("range_oob", "positive ranges"),
        ("range_empty", "positive ranges"),
        ("range_reversed", "positive ranges"),
        ("range_float", "integers"),
        ("mapping_dtype", "1D int32/int64"),
        ("mapping_rank", "1D int32/int64"),
        ("mapping_device", "1D int32/int64"),
        ("stream_device", "transfer stream"),
        ("direction", "direction must be bool"),
        ("group_type", "kv_group"),
    ],
)
def test_validation_fails_before_native_launch(row_env, case, error):
    env = row_env
    planes = list(env.planes(0))
    chunks = _chunks(planes, [2, 1])
    starts, ends = [10, 12], [12, 13]
    slots = torch.tensor([2, 4, 6])
    kwargs = dict(kv_group=0, direction=True, slot_mapping_base=10)
    layout = env.connector._group_layouts[0]
    if case == "missing_group":
        del env.connector._group_layouts[0]
    elif case == "empty_layout":
        layout.num_layers = 0
    elif case == "layout_indices":
        layout.layer_indices = (0,)
    elif case == "layout_dims":
        layout.k_hidden_dims = 0
    elif case == "format":
        layout.kv_format = KVCacheFormat.SEPARATE_KV
    elif case == "two_major":
        layout.vllm_two_major = True
    elif case == "layout_device":
        layout.kv_device = torch.device("cpu")
    elif case == "connector_dtype":
        env.connector.dtype = torch.float16
    elif case == "plane_count":
        planes.pop()
    elif case == "plane_object":
        planes[0] = object()
    elif case == "plane_dtype":
        planes[0] = planes[0].float()
    elif case == "plane_cpu":
        planes[0] = planes[0].as_subclass(torch.Tensor)
    elif case == "plane_device":
        planes[1] = planes[1].as_subclass(_OtherNPUTensor)
    elif case == "plane_rank":
        planes[0] = planes[0].squeeze().flatten(2)
    elif case == "plane_stride":
        planes[0] = planes[0].transpose(0, 1)
    elif case == "plane_width":
        planes[0] = planes[0].view(3, 4, 1, 64)
    elif case == "plane_blocks":
        planes[1] = planes[1][:2]
    elif case == "plane_zero":
        planes[0] = planes[0][:0]
    elif case == "chunk_count":
        chunks.pop()
    elif case == "chunk_short":
        chunks[0] = chunks[0][:-1]
    elif case == "chunk_padded_tail":
        chunks[1] = torch.empty_like(chunks[0])
    elif case == "chunk_stride":
        chunks[0] = torch.empty(chunks[0].numel() * 2, dtype=torch.bfloat16)[::2]
    elif case == "chunk_dtype":
        chunks[0] = chunks[0].float()
    elif case == "chunk_npu":
        chunks[0] = chunks[0].as_subclass(_NPUTensor)
    elif case == "chunk_object":
        chunks[0] = SimpleNamespace(tensor=chunks[0])
    elif case == "base_negative":
        kwargs["slot_mapping_base"] = -1
    elif case == "base_float":
        kwargs["slot_mapping_base"] = 10.0
    elif case == "range_before_base":
        starts[0] = 9
    elif case == "range_oob":
        ends[-1] = 14
    elif case == "range_empty":
        ends[0] = starts[0]
    elif case == "range_reversed":
        ends[0] = starts[0] - 1
    elif case == "range_float":
        starts[0] = 10.0
    elif case == "mapping_dtype":
        slots = slots.float()
    elif case == "mapping_rank":
        slots = slots.unsqueeze(0)
    elif case == "mapping_device":
        slots = slots.as_subclass(_OtherNPUTensor)
    elif case == "stream_device":
        env.store.device = _OTHER_DEVICE
    elif case == "direction":
        kwargs["direction"] = 1
    elif case == "group_type":
        kwargs["kv_group"] = True
    with pytest.raises(ValueError, match=error):
        env.connector.transfer_layerwise_prefill_row(
            planes, chunks, starts, ends, slots, **kwargs
        )
    assert not env.calls and not env.events


@pytest.mark.parametrize("failure", [0, -1, "null", "raise"])
def test_registered_pointer_failure_is_closed_and_fenced(row_env, failure):
    env = row_env
    env.pointer_error = failure
    planes = env.planes(1)
    with pytest.raises(RuntimeError, match="registered|registration"):
        env.connector.transfer_layerwise_prefill_row(
            planes,
            _chunks(planes, [1]),
            [0],
            [1],
            torch.tensor([0]),
            kv_group=1,
            direction=False,
        )
    assert not env.calls
    assert env.events[-1] == ("load", "sync")


@pytest.mark.parametrize("direction", [True, False])
def test_launch_failure_still_fences_before_releasing_owners(row_env, direction):
    env = row_env
    env.launch_error = True
    with pytest.raises(RuntimeError, match="native launch failed"):
        env.connector.transfer_layerwise_prefill_row(
            env.planes(1),
            _chunks(env.planes(1), [1]),
            [0],
            [1],
            torch.tensor([0]),
            kv_group=1,
            direction=direction,
        )
    assert not env.pending
    assert env.events[-1] == ("store" if direction else "load", "sync")
    assert getattr(env.connector, "_layerwise_prefill_row_failed_owners", None) is None


def test_fence_failure_retains_raw_pointer_owners_and_disables_hook(row_env):
    env = row_env
    env.sync_error = True

    def fail_transfer():
        with pytest.raises(RuntimeError, match="completion failed"):
            env.connector.transfer_layerwise_prefill_row(
                env.planes(1),
                _chunks(env.planes(1), [1]),
                [0],
                [1],
                torch.tensor([0]),
                kv_group=1,
                direction=True,
            )

    fail_transfer()
    gc.collect()
    assert all(ref() is not None for ref in env.last_refs)
    assert env.connector._layerwise_prefill_row_failed_owners
    with pytest.raises(RuntimeError, match="previous prefill row completion fence"):
        env.connector.transfer_layerwise_prefill_row(
            (), (), (), (), None, kv_group=1, direction=True
        )
    assert len(env.calls) == 1
    env.sync_error = False
    env.store.synchronize()


def test_explicit_zero_work_does_not_initialize_or_launch(row_env):
    env = row_env
    env.connector._group_layouts.clear()
    assert (
        env.connector.transfer_layerwise_prefill_row(
            (), (), (), (), None, kv_group=1, direction=False
        )
        is None
    )
    assert not env.events and not env.calls


@pytest.mark.parametrize("fmt", [KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV])
def test_registered_legacy_mla_dsa_planes(row_env, fmt):
    env = row_env
    planes = env.planes(0)
    if fmt == KVCacheFormat.DSA_KV:
        planes += env.planes(1)
    env.connector.dsa_two_groups = False
    env.connector._group_layouts.clear()
    VLLMPagedMemLayerwiseNPUConnector._lazy_initialize_buffer(
        env.connector, [planes] * 79, kv_group=0, init_staging=False
    )
    env.connector.transfer_layerwise_prefill_row(
        planes,
        _chunks(planes, [2, 1]),
        [0, 2],
        [2, 3],
        torch.tensor([11, 0, 3]),
        kv_group=0,
        direction=True,
    )
    assert env.calls[0]["fmt"] == fmt.value
    assert env.calls[0]["widths"] == (64, 16, 32 if fmt == KVCacheFormat.DSA_KV else 0)
    assert len(env.calls[0]["plane_ptrs"]) == len(planes)


def test_metadata_creation_failure_is_fenced_without_launch(row_env, monkeypatch):
    env = row_env
    factory = torch.tensor

    def fail_offsets(*args, **kwargs):
        if kwargs.get("device") is _DEVICE and kwargs.get("dtype") == torch.int32:
            raise RuntimeError("metadata allocation failed")
        return factory(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", fail_offsets)
    planes = env.planes(1)
    with pytest.raises(RuntimeError, match="metadata allocation failed"):
        env.connector.transfer_layerwise_prefill_row(
            planes,
            _chunks(planes, [1]),
            [0],
            [1],
            torch.tensor([0]),
            kv_group=1,
            direction=False,
        )
    assert not env.calls
    assert env.events[-1] == ("load", "sync")


def test_success_does_not_cache_request_tensor_owners(row_env):
    env = row_env
    env.connector.transfer_layerwise_prefill_row(
        env.planes(1),
        _chunks(env.planes(1), [1]),
        [0],
        [1],
        torch.tensor([0]),
        kv_group=1,
        direction=True,
    )
    gc.collect()
    assert all(ref() is None for ref in env.last_refs)
    assert getattr(env.connector, "_layerwise_prefill_row_failed_owners", None) is None


def test_four_prepared_mappings_512_chunks_101_rows_no_repacking_or_readback(row_env):
    env = row_env
    sizes = [1] * 511 + [3]
    total, capacity = sum(sizes), 516
    starts = [100 + i for i in range(512)]
    ends = [start + size for start, size in zip(starts, sizes, strict=True)]
    banks = {}
    for group in (0, 1):
        for bank in (0, 1):
            planes = env.planes(group, blocks=capacity // 4)
            for plane in planes:
                plane.copy_(torch.arange(plane.numel()).reshape(plane.shape) % 251)
            chunks = _chunks(planes, sizes)
            if bank:
                for index, chunk in enumerate(chunks):
                    chunk.copy_((torch.arange(chunk.numel()) + index) % 127)
            mapping = torch.arange(total, dtype=torch.int32).roll(17 + bank)
            plan = env.connector.prepare_layerwise_prefill_slots(
                mapping, kv_group=group, capacity=capacity
            )
            banks[group, bank] = (
                planes,
                chunks,
                mapping,
                plan,
                [p.as_subclass(torch.Tensor).clone() for p in planes],
            )
    assert env.host_ops["uploads"] == env.host_ops["tolist"] == 4
    assert env.host_ops["unique_checks"] == 4
    assert env.host_ops["validation_scans"] == 8
    assert env.host_ops["readback"] == env.host_ops["cat"] == 0
    before = env.host_ops.copy()

    for group, row_count in ((0, 79), (1, 22)):
        for row in range(row_count):
            bank = row % 2
            planes, chunks, mapping, plan, _ = banks[group, bank]
            env.connector.transfer_layerwise_prefill_row(
                planes,
                chunks,
                starts,
                ends,
                plan,
                kv_group=group,
                direction=not bank,
                slot_mapping_base=100,
            )
            assert env.calls[-1]["slot_ptr"] == plan._slots.data_ptr()
            assert env.calls[-1]["plane_ptrs"] == [p.data_ptr() for p in planes]
            assert env.calls[-1]["chunk_ptrs"] == [c.data_ptr() for c in chunks]

    assert len(env.calls) == 101 and not env.pending
    assert env.host_ops == {**before, "slices": before["slices"] + 101}
    assert sum(event[1] == "registered" for event in env.events) == 101 * 512
    assert sum(event[1] == "sync" for event in env.events) == 101
    for (group, bank), (planes, chunks, mapping, _, snapshots) in banks.items():
        cursor = 0
        expected = [snapshot.view(capacity, -1).clone() for snapshot in snapshots]
        for chunk, size in zip(chunks, sizes, strict=True):
            plane_offset = 0
            selected = mapping[cursor : cursor + size].long()
            for reference in expected:
                width = reference.shape[1]
                packed = chunk[plane_offset : plane_offset + size * width].view(
                    size, width
                )
                if bank:
                    reference[selected] = packed
                else:
                    torch.testing.assert_close(
                        packed, reference[selected], rtol=0, atol=0
                    )
                plane_offset += size * width
            cursor += size
        for plane, reference in zip(planes, expected, strict=True):
            torch.testing.assert_close(
                plane.as_subclass(torch.Tensor).view(capacity, -1),
                reference,
                rtol=0,
                atol=0,
            )
    assert getattr(env.connector, "_layerwise_prefill_row_failed_owners", None) is None


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("on_npu", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_consecutive_noncontiguous_mapping_and_inference_snapshot(
    row_env, dtype, on_npu, prepared
):
    env = row_env
    planes = env.planes(1)
    chunks = _chunks(planes, [2, 1])
    with torch.inference_mode():
        source = torch.tensor([11, -1, 4, -1, 0, -1, 9, -1], dtype=dtype)[::2]
        if on_npu:
            source = source.as_subclass(_NPUTensor)
        mapping = source
        if prepared:
            mapping = env.connector.prepare_layerwise_prefill_slots(
                source, kv_group=1, capacity=12
            )
            assert not torch.is_inference(mapping._slots)
            assert mapping._slots._version == mapping._version
            assert mapping._slots.dtype == torch.int64
            assert mapping._slots.device == _DEVICE
            assert mapping._slots.is_contiguous()
            assert mapping._slots.data_ptr() != source.data_ptr()
            source.fill_(-1)
        before = env.host_ops.copy()
        env.connector.transfer_layerwise_prefill_row(
            planes,
            chunks,
            [101, 103],
            [103, 104],
            mapping,
            kv_group=1,
            direction=False,
            slot_mapping_base=100,
        )
    assert env.calls[0]["slots"] == [4, 0, 9]
    assert env.host_ops["cat"] == before["cat"]
    assert env.host_ops["slices"] == before["slices"] + 1
    assert env.host_ops["tolist"] == before["tolist"] + (not prepared)
    assert env.host_ops["unique_checks"] == before["unique_checks"] + (not prepared)
    if prepared:
        assert env.calls[0]["slot_ptr"] == mapping._slots.data_ptr() + 8
        assert env.host_ops["readback"] == before["readback"]
        assert env.host_ops["uploads"] == before["uploads"]


@pytest.mark.parametrize(
    "case, error",
    [
        ("missing_group", "registered full group"),
        ("empty_layout", "registered full group"),
        ("layout_indices", "registered full group"),
        ("layout_device", "NPU layout"),
        ("group_type", "kv_group"),
        ("group_negative", "kv_group"),
        ("capacity_zero", "capacity"),
        ("capacity_negative", "capacity"),
        ("capacity_float", "capacity"),
        ("capacity_bool", "capacity"),
        ("mapping_object", "1D int32/int64"),
        ("mapping_dtype", "1D int32/int64"),
        ("mapping_rank", "1D int32/int64"),
        ("mapping_device", "1D int32/int64"),
        ("negative_slot", "capacity"),
        ("oob_slot", "capacity"),
        ("duplicate", "duplicate"),
    ],
)
def test_prepare_slots_rejects_invalid_mapping_or_binding(row_env, case, error):
    env = row_env
    mapping = torch.tensor([0, 4, 11])
    kwargs = dict(kv_group=1, capacity=12)
    layout = env.connector._group_layouts[1]
    if case == "missing_group":
        del env.connector._group_layouts[1]
    elif case == "empty_layout":
        layout.num_layers = 0
    elif case == "layout_indices":
        layout.layer_indices = (0,)
    elif case == "layout_device":
        layout.kv_device = torch.device("cpu")
    elif case == "group_type":
        kwargs["kv_group"] = True
    elif case == "group_negative":
        kwargs["kv_group"] = -1
    elif case.startswith("capacity_"):
        kwargs["capacity"] = dict(zero=0, negative=-1, float=12.0, bool=True)[
            case.removeprefix("capacity_")
        ]
    elif case == "mapping_object":
        mapping = [0, 4, 11]
    elif case == "mapping_dtype":
        mapping = mapping.float()
    elif case == "mapping_rank":
        mapping = mapping.unsqueeze(0)
    elif case == "mapping_device":
        mapping = mapping.as_subclass(_OtherNPUTensor)
    elif case == "negative_slot":
        mapping[0] = -1
    elif case == "oob_slot":
        mapping[-1] = 12
    elif case == "duplicate":
        mapping[-1] = mapping[0]
    with pytest.raises(ValueError, match=error):
        env.connector.prepare_layerwise_prefill_slots(mapping, **kwargs)
    assert not env.calls and not env.events


@pytest.mark.parametrize(
    "case, error",
    [
        ("connector", "another connector"),
        ("group", "another group/layout"),
        ("layout", "another group/layout"),
        ("capacity", "capacity"),
        ("device", "another NPU device"),
        ("mutation", "mutated"),
        ("inference_mutation", "mutated"),
        ("resize", "mutated"),
        ("range_oob", "positive ranges"),
        ("chunk_size", "exact packed"),
    ],
)
def test_prepared_slots_reuse_checks_binding_version_and_row(row_env, case, error):
    env = row_env
    plan = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([4, 0, 9]), kv_group=1, capacity=12
    )
    connector, group = env.connector, 1
    planes = env.planes(group)
    chunks = _chunks(planes, [2, 1])
    ends = [2, 3]
    if case == "connector":
        connector = copy.copy(connector)
    elif case == "group":
        group, planes = 0, env.planes(0)
    elif case == "layout":
        connector._group_layouts[1] = copy.copy(connector._group_layouts[1])
    elif case == "capacity":
        planes = tuple(p[:2] for p in planes)
    elif case == "device":
        connector._group_layouts[1].kv_device = _OTHER_DEVICE
        planes = tuple(p.as_subclass(_OtherNPUTensor) for p in planes)
    elif case == "mutation":
        plan._slots[0] = 4  # Even a same-value write invalidates the snapshot.
    elif case == "inference_mutation":
        with torch.inference_mode():
            plan._slots[0] = -1
    elif case == "resize":
        plan._slots.resize_(2)
    elif case == "range_oob":
        ends[-1] = 4
    elif case == "chunk_size":
        chunks[0] = chunks[0][:-1]
    env.events.clear()
    with pytest.raises(ValueError, match=error):
        connector.transfer_layerwise_prefill_row(
            planes, chunks, [0, 2], ends, plan, kv_group=group, direction=False
        )
    assert not env.events and not env.calls


def test_prepared_plan_fields_cannot_be_rebound(row_env):
    plan = row_env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([0]), kv_group=1, capacity=12
    )
    for field, value in (
        ("_slots", torch.tensor([-1])),
        ("_version", -1),
        ("_owner", object()),
        ("_kv_group", 0),
        ("_capacity", 24),
        ("_length", 2),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(plan, field, value)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("direction", [False, True])
@pytest.mark.parametrize(
    "starts, ends, selected, duplicate",
    [
        ([0, 3], [2, 5], [4, 0, 2, 7], False),
        ([3, 0], [5, 2], [2, 7, 4, 0], False),
        ([0, 1], [2, 3], [4, 0, 0, 9], True),
    ],
)
def test_generic_ranges_preserve_order_gaps_and_overlap_checks(
    row_env, prepared, direction, starts, ends, selected, duplicate
):
    env = row_env
    planes = env.planes(1)
    mapping = torch.tensor([4, 0, 9, 2, 7])
    if prepared:
        mapping = env.connector.prepare_layerwise_prefill_slots(
            mapping, kv_group=1, capacity=12
        )
    before = env.host_ops.copy()
    args = (planes, _chunks(planes, [2, 2]), starts, ends, mapping)
    if duplicate and not direction:
        with pytest.raises(ValueError, match="duplicate"):
            env.connector.transfer_layerwise_prefill_row(
                *args, kv_group=1, direction=direction
            )
        assert not env.calls
    else:
        env.connector.transfer_layerwise_prefill_row(
            *args, kv_group=1, direction=direction
        )
        assert env.calls[0]["slots"] == selected
    assert env.host_ops["cat"] == before["cat"] + 1
    assert env.host_ops["tolist"] == before["tolist"] + 1
    assert env.events[-1] == ("store" if direction else "load", "sync")


@pytest.mark.parametrize("failure", [None, "launch", "fence"])
def test_prepared_plan_owners_survive_errors_without_success_cache(row_env, failure):
    env = row_env
    env.launch_error = failure == "launch"
    env.sync_error = failure == "fence"
    plan_refs = []

    def transfer():
        plan = env.connector.prepare_layerwise_prefill_slots(
            torch.tensor([0]), kv_group=1, capacity=12
        )
        plan_refs.extend((weakref.ref(plan), weakref.ref(plan._slots)))
        env.connector.transfer_layerwise_prefill_row(
            env.planes(1),
            _chunks(env.planes(1), [1]),
            [0],
            [1],
            plan,
            kv_group=1,
            direction=True,
        )

    if failure:
        with pytest.raises(
            RuntimeError, match="completion failed|native launch failed"
        ):
            transfer()
    else:
        transfer()
    gc.collect()
    if failure == "fence":
        assert all(ref() is not None for ref in plan_refs + env.last_refs)
        # Preparation and even zero-work transfers must not mask a prior failure.
        with pytest.raises(RuntimeError, match="previous prefill row completion fence"):
            env.connector.prepare_layerwise_prefill_slots(None, kv_group=-1, capacity=0)
        with pytest.raises(RuntimeError, match="previous prefill row completion fence"):
            env.connector.transfer_layerwise_prefill_row(
                (), (), (), (), plan_refs[0](), kv_group=1, direction=True
            )
        env.sync_error = False
        env.store.synchronize()
    else:
        assert all(ref() is None for ref in plan_refs + env.last_refs)
        assert not env.pending
        assert (
            getattr(env.connector, "_layerwise_prefill_row_failed_owners", None) is None
        )


def test_optional_host_timings_accumulate_without_extra_sync(row_env, monkeypatch):
    env = row_env
    planes = env.planes(1)
    plan = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([0]), kv_group=1, capacity=12
    )
    args = (planes, _chunks(planes, [1]), [0], [1], plan)

    def forbidden():
        pytest.fail("Timing disabled must not read the clock")

    monkeypatch.setattr(npu_connectors, "perf_counter", forbidden)
    env.connector.transfer_layerwise_prefill_row(*args, kv_group=1, direction=True)
    clock = iter([0.0, 1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 16.0])
    monkeypatch.setattr(npu_connectors, "perf_counter", lambda: next(clock))
    timing = dict(prepare_s=10.0, submit_s=20.0, fence_s=30.0, other=7.0)
    for _ in range(2):
        assert (
            env.connector.transfer_layerwise_prefill_row(
                *args, kv_group=1, direction=True, timing=timing
            )
            is None
        )
    assert env.connector.supports_layerwise_prefill_transfer_timings is True
    assert timing == dict(prepare_s=12.0, submit_s=24.0, fence_s=36.0, other=7.0)
    assert sum(event[1] == "sync" for event in env.events) == 3


@pytest.mark.parametrize("failure", ["registration", "launch", "fence"])
def test_host_timings_include_failed_submission_and_fence(
    row_env, monkeypatch, failure
):
    env = row_env
    planes = env.planes(1)
    plan = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([0]), kv_group=1, capacity=12
    )
    env.pointer_error = "raise" if failure == "registration" else None
    env.launch_error = failure == "launch"
    env.sync_error = failure == "fence"
    clock = iter([0.0, 1.0, 3.0] if failure == "registration" else [0.0, 1.0, 3.0, 6.0])
    monkeypatch.setattr(npu_connectors, "perf_counter", lambda: next(clock))
    timing = {}
    with pytest.raises(RuntimeError, match="failed"):
        env.connector.transfer_layerwise_prefill_row(
            planes,
            _chunks(planes, [1]),
            [0],
            [1],
            plan,
            kv_group=1,
            direction=True,
            timing=timing,
        )
    assert timing == dict(
        prepare_s=1.0,
        submit_s=0.0 if failure == "registration" else 2.0,
        fence_s=2.0 if failure == "registration" else 3.0,
    )
    assert env.events[-1] == ("store", "sync")
    if failure == "fence":
        env.sync_error = False
        env.store.synchronize()
