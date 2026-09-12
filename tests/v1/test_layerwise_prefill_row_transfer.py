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
        self.queue = []
        self.completed = 0

    def run_to(self, mark):
        while self.completed < mark:
            operation = self.queue[self.completed]
            operation()
            self.queue[self.completed] = None
            self.completed += 1

    def wait_stream(self, producer):
        assert self.env.active is self
        self.env.events.append((self.name, "wait", producer.name))
        # torch_npu Stream.wait_stream calls producer.record_event(), which
        # allocates a Python event and lazily creates its native handle.
        event = torch.npu.Event()
        event.record(producer)
        self.wait_event(event)

    def wait_event(self, event):
        assert self.env.active is self or self is self.env.compute
        assert event.stream is not None
        self.env.events.append((self.name, "wait_event", event))
        stream, mark = event.stream, event.mark
        self.queue.append(lambda: stream.run_to(mark))

    def synchronize(self):
        self.env.events.append((self.name, "sync"))
        if self.env.sync_error:
            raise RuntimeError("completion failed")
        self.run_to(len(self.queue))


class _Event:
    def __init__(self, env):
        self.env = env
        self.stream = None
        self.mark = None
        self.materialized = False
        env.events.append(("event", "allocate", self))

    def record(self, stream):
        if not self.materialized:
            assert not self.env.pre_hcom, "Native event allocation in pre-HCOM"
            self.env.events.append(("event", "materialize", self))
            self.materialized = True
        self.env.events.append((stream.name, "record", self))
        if self.env.event_record_error is self or (
            self.env.event_record_error is True and stream is not self.env.compute
        ):
            raise RuntimeError("event record failed")
        self.stream, self.mark = stream, len(stream.queue)

    def synchronize(self):
        self.env.events.append((self.stream.name, "event_sync", self))
        if self.env.sync_error or self.env.event_sync_error:
            raise RuntimeError("event completion failed")
        self.stream.run_to(self.mark)


@pytest.fixture
def row_env(monkeypatch):
    env = SimpleNamespace(
        events=[],
        calls=[],
        pending=[],
        uploads=[],
        active=None,
        pre_hcom=False,
        launch_error=False,
        sync_error=False,
        event_record_error=False,
        event_sync_error=False,
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
    monkeypatch.setattr(torch.npu, "Event", lambda: _Event(env))
    tensor_factory, tensor_to, tensor_cat = torch.tensor, torch.Tensor.to, torch.cat
    tensor_empty, tensor_copy = torch.empty, torch.Tensor.copy_
    tensor_is_pinned = torch.Tensor.is_pinned
    tensor_tolist, tensor_getitem = torch.Tensor.tolist, torch.Tensor.__getitem__

    def upload(dst, src, non_blocking=False):
        stream = env.active or env.compute
        call = dict(
            src=weakref.ref(src),
            dst=weakref.ref(dst),
            pinned=src.is_pinned(),
            non_blocking=non_blocking,
            nbytes=src.numel() * src.element_size(),
            stream=stream,
            complete=False,
        )
        env.uploads.append(call)
        env.host_ops["uploads"] += 1
        # Version increments at enqueue, not when the device executes the DMA.
        torch.autograd.graph.increment_version(dst)

        def dma():
            assert call["src"]() is not None and call["dst"]() is not None
            tensor_copy(call["dst"]().as_subclass(torch.Tensor).data, call["src"]())
            call["complete"] = True

        stream.queue.append(dma)
        if not (non_blocking and call["pinned"]):
            # CopyKernel synchronizes CURRENT stream, including older KV work.
            # CachingHostAllocator does the same for pageable nonblocking copies.
            env.events.append((stream.name, "copy_sync"))
            stream.run_to(len(stream.queue))
        return dst

    def empty_tensor(*args, **kwargs):
        on_npu = kwargs.get("device") is _DEVICE
        pinned = kwargs.pop("pin_memory", False)
        if on_npu:
            assert env.active is not None
            assert kwargs["dtype"] in (torch.int32, torch.int64)
            kwargs["device"] = "cpu"
        result = tensor_empty(*args, **kwargs)
        if on_npu:
            result.fill_(-999)  # Detect metadata inspected before deferred DMA.
            return result.as_subclass(_NPUTensor)
        result._mock_pinned = pinned
        return result

    def make_tensor(*args, **kwargs):
        if kwargs.get("device") is _DEVICE:
            assert env.active is not None
            assert kwargs["dtype"] in (torch.int32, torch.int64)
            env.events.append((env.active.name, "metadata", kwargs["dtype"]))
            kwargs["device"] = "cpu"
            src = tensor_factory(*args, **kwargs)
            dst = tensor_empty(src.shape, dtype=src.dtype).as_subclass(_NPUTensor)
            return upload(dst, src)
        pinned = kwargs.pop("pin_memory", False)
        result = tensor_factory(*args, **kwargs)
        result._mock_pinned = pinned
        return result

    def copy_tensor(dst, src, non_blocking=False):
        if (
            isinstance(dst, _NPUTensor)
            and dst.dtype in (torch.int32, torch.int64)
            and src.device.type == "cpu"
        ):
            return upload(dst, src, non_blocking)
        return tensor_copy(dst, src, non_blocking=non_blocking)

    def is_pinned(tensor, *args, **kwargs):
        return getattr(tensor, "_mock_pinned", False) or tensor_is_pinned(
            tensor, *args, **kwargs
        )

    def to_tensor(tensor, *args, **kwargs):
        if kwargs.get("device") is _DEVICE:
            assert tensor.dtype in (torch.int32, torch.int64), "KV staging forbidden"
            env.events.append(((env.active or env.compute).name, "slots"))
            kwargs["device"] = "cpu"
            non_blocking = kwargs.pop("non_blocking", False)
            tensor = tensor.as_subclass(torch.Tensor)
            src = tensor_to(tensor, *args, **kwargs)
            dst = tensor_empty(src.shape, dtype=src.dtype).as_subclass(_NPUTensor)
            return upload(dst, src, non_blocking)
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
    monkeypatch.setattr(torch, "empty", empty_tensor)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
    monkeypatch.setattr(torch.Tensor, "is_pinned", is_pinned)
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
        env.events.append((env.active.name, "launch"))
        call = dict(
            slot_ptr=slots.data_ptr(),
            total=total,
            fmt=fmt,
            widths=(k_width, v_width, dsa_width),
            direction=direction,
            plane_ptrs=[plane.data_ptr() for plane in planes],
            chunk_ptrs=[chunk.data_ptr() for chunk in chunks],
        )
        env.calls.append(call)
        # The native ABI only owns raw pointers. Do not let the mock hide a
        # premature Python-owner release by holding strong tensor references.
        plane_refs = [weakref.ref(t) for t in planes]
        chunk_refs = [weakref.ref(t) for t in chunks]
        metadata_refs = [weakref.ref(t) for t in (slots, offsets, sizes, pointers)]
        env.last_refs = plane_refs + chunk_refs + metadata_refs

        def complete():
            assert all(
                ref() is not None for ref in plane_refs + chunk_refs + metadata_refs
            )
            slot_values = tensor_tolist(metadata_refs[0]().as_subclass(torch.Tensor))
            offset_values = tensor_tolist(metadata_refs[1]().as_subclass(torch.Tensor))
            size_values = tensor_tolist(metadata_refs[2]().as_subclass(torch.Tensor))
            assert tensor_tolist(metadata_refs[3]().as_subclass(torch.Tensor)) == [
                ref().data_ptr() + 4096 for ref in chunk_refs
            ]
            call.update(slots=slot_values, offsets=offset_values, sizes=size_values)
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
            env.pending.remove(complete)

        env.pending.append(complete)
        env.active.queue.append(complete)
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
@pytest.mark.parametrize("asynchronous", [False, True])
def test_validation_fails_before_native_launch(row_env, case, error, asynchronous):
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
    method = (
        env.connector.prepare_layerwise_prefill_row
        if asynchronous
        else env.connector.transfer_layerwise_prefill_row
    )
    with pytest.raises(ValueError, match=error):
        method(planes, chunks, starts, ends, slots, **kwargs)
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
    assert env.host_ops == {
        **before,
        "slices": before["slices"] + 101,
        "uploads": before["uploads"] + 303,
    }
    assert all(not upload["non_blocking"] for upload in env.uploads)
    # The chunk memo resolves each distinct (mapping, chunk) tensor once; the
    # same chunk objects recur across all 101 rows, so only 4 x 512 first-time
    # registrations remain instead of one per row.
    assert sum(event[1] == "registered" for event in env.events) == 4 * 512
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
        assert env.host_ops["uploads"] == before["uploads"] + 3


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


def _prepare_async(env, *, group=1, direction=True, slots=None):
    planes = env.planes(group)
    return env.connector.prepare_layerwise_prefill_row(
        planes,
        _chunks(planes, [2, 1]),
        [100, 102],
        [102, 103],
        torch.tensor([4, 0, 9]) if slots is None else slots,
        kv_group=group,
        direction=direction,
        slot_mapping_base=100,
    )


@pytest.mark.parametrize("older_group", [0, 1])
def test_async_prepare_latent_does_not_drain_gated_older_load(row_env, older_group):
    env = row_env
    plans = {
        group: env.connector.prepare_layerwise_prefill_slots(
            torch.tensor([4, 0, 9]), kv_group=group, capacity=12
        )
        for group in (0, 1)
    }
    older = _prepare_async(
        env, group=older_group, direction=False, slots=plans[older_group]
    )
    released = False

    def gate():
        assert released, "Metadata upload drained gated older KV"

    env.load.queue.append(gate)
    env.connector.submit_layerwise_prefill_row(older)
    completed = env.load.completed
    before = env.host_ops.copy()
    later = _prepare_async(env, group=0, direction=False, slots=plans[0])
    assert env.load.completed == completed and len(env.pending) == 1
    assert env.host_ops == {
        **before,
        "slices": before["slices"] + 1,
        "uploads": before["uploads"] + 3,
    }
    assert all(
        upload["pinned"] and upload["non_blocking"] and not upload["complete"]
        for upload in env.uploads[-3:]
    )
    assert sum(upload["nbytes"] for upload in env.uploads[-3:]) == 2 * 16
    chunks, planes = later._args[:2]
    chunks[0].fill_(21)
    chunks[1].fill_(22)
    env.compute.queue.append(lambda: [plane.fill_(31) for plane in planes])
    env.pre_hcom = True
    env.connector.submit_layerwise_prefill_row(later)
    env.connector.wait_layerwise_prefill_row(later)
    env.pre_hcom = False
    assert env.load.completed == completed and len(env.pending) == 2
    assert "sizes" not in env.calls[-1]  # Native reads metadata only at execution.
    released = True
    env.compute.run_to(len(env.compute.queue))
    for plane in planes:
        flat = plane.as_subclass(torch.Tensor).view(12, -1)
        assert bool((flat[[4, 0]] == 21).all())
        assert bool((flat[9] == 22).all())
        assert bool((flat[1] == 31).all())
    assert env.calls[-1]["slots"] == [4, 0, 9]
    assert env.calls[-1]["sizes"] == [2, 1]
    env.connector.drain_layerwise_prefill_transfers()


@pytest.mark.parametrize(
    "method, pinned, non_blocking",
    [
        ("factory", False, False),
        ("to", False, False),
        ("copy", False, False),
        ("copy", False, True),
        ("copy", True, False),
        ("copy", True, True),
    ],
)
def test_mock_metadata_copy_models_torch_npu_current_stream_sync(
    row_env, method, pinned, non_blocking
):
    env = row_env
    payload = []
    env.load.queue.append(lambda: payload.append("older KV"))
    with torch.npu.stream(env.load):
        host = torch.tensor([1, 2], dtype=torch.int32, pin_memory=pinned)
        if method == "factory":
            dst = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
        elif method == "to":
            dst = host.to(device=_DEVICE, non_blocking=non_blocking)
        else:
            dst = torch.empty(2, dtype=torch.int32, device=_DEVICE)
            dst.copy_(host, non_blocking=non_blocking)
    assert bool(payload) is not (pinned and non_blocking)
    env.load.run_to(len(env.load.queue))
    torch.testing.assert_close(dst.as_subclass(torch.Tensor), host)


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("direction", [False, True])
@pytest.mark.parametrize("prepared_slots", [False, True])
def test_async_enqueue_has_no_allocation_readback_or_host_fence(
    row_env,
    monkeypatch,
    group,
    direction,
    prepared_slots,
):
    env = row_env
    slots = torch.tensor([4, 0, 9])
    if prepared_slots:
        slots = env.connector.prepare_layerwise_prefill_slots(
            slots,
            kv_group=group,
            capacity=12,
        )
    row = _prepare_async(env, group=group, direction=direction, slots=slots)
    chunks, planes = row._args[:2]
    snapshots = [p.as_subclass(torch.Tensor).clone() for p in planes]
    assert not env.calls and not env.pending
    assert not env.connector._layerwise_prefill_inflight
    assert not any(e[1] in ("sync", "event_sync") for e in env.events)
    assert env.connector.supports_layerwise_prefill_async_rows is True
    assert row._producer_event is not row._done_event
    assert row._producer_event.materialized and row._done_event.materialized
    uploads = env.uploads[-3:]
    for upload in uploads:
        assert upload["pinned"] and upload["non_blocking"]
        assert not upload["complete"]
        for name in ("src", "dst"):
            assert any(owner is upload[name]() for owner in row._owners)
            assert any(t is upload[name]() for t, _ in row._metadata_versions)
    if prepared_slots:
        assert not any(e[1] == "copy_sync" for e in env.events if e[0] != "compute")
    before = env.host_ops.copy()
    env.events.clear()
    native = npu_connectors.lmc_ops.dense_mla_dsa_batched_direct_kv_transfer

    def launch(*args):
        assert env.connector._layerwise_prefill_inflight[id(row)] is row
        assert row._state == "submitted"
        native(*args)

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Pre-HCOM submit/wait performed allocation, validation or host sync"
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            npu_connectors.lmc_ops, "dense_mla_dsa_batched_direct_kv_transfer", launch
        )
        for name in ("tensor", "empty", "empty_like", "zeros", "ones", "cat"):
            patch.setattr(torch, name, forbidden)
        for name in ("clone", "to", "cpu", "tolist", "item", "__getitem__", "copy_"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch.npu, "Event", forbidden)
        patch.setattr(_Stream, "synchronize", forbidden)
        patch.setattr(_Event, "synchronize", forbidden)
        patch.setattr(npu_connectors, "perf_counter", forbidden)
        patch.setattr(env.connector, "_validate_layerwise_prefill_row", forbidden)
        patch.setattr(npu_connectors.lmc_ops, "get_device_ptr", forbidden)
        env.pre_hcom = True
        ticket = env.connector.submit_layerwise_prefill_row(row)
        assert ticket is row
        assert env.connector.wait_layerwise_prefill_row(ticket) is None
        env.pre_hcom = False
    assert env.host_ops == before
    stream = "store" if direction else "load"
    assert env.events == [
        (stream, "enter"),
        ("compute", "record", row._producer_event),
        (stream, "wait_event", row._producer_event),
        (stream, "launch"),
        (stream, "record", ticket.done_event),
        (stream, "exit"),
        ("compute", "wait_event", ticket.done_event),
    ]
    assert len(env.pending) == 1
    for chunk in chunks:
        assert bool((chunk == -7).all())
    for plane, snapshot in zip(planes, snapshots, strict=True):
        torch.testing.assert_close(plane.as_subclass(torch.Tensor), snapshot)
    assert all(ref() is not None for ref in env.last_refs)
    assert env.connector.complete_layerwise_prefill_row(ticket) is None
    assert all(upload["complete"] for upload in uploads)
    assert not env.pending and not env.connector._layerwise_prefill_inflight
    assert env.events[-1] == (stream, "event_sync", ticket.done_event)
    assert not ticket._args and not ticket._owners and not ticket._tensor_identities
    for plane, snapshot in zip(planes, snapshots, strict=True):
        flat = plane.as_subclass(torch.Tensor).view(12, -1)
        if direction:
            torch.testing.assert_close(plane.as_subclass(torch.Tensor), snapshot)
        else:
            assert bool((flat[[4, 0, 9]] == -7).all())
    if direction:
        for chunk, selected in zip(chunks, ([4, 0], [9]), strict=True):
            offset = 0
            for snapshot in snapshots:
                expected = snapshot.view(12, -1)[selected].flatten()
                torch.testing.assert_close(
                    chunk[offset : offset + expected.numel()], expected
                )
                offset += expected.numel()


@pytest.mark.parametrize("switch_current", [False, True])
def test_async_post_sfa_dependency_and_compute_consumer(row_env, switch_current):
    env = row_env
    row = _prepare_async(env, direction=False)
    chunks, planes = row._args[:2]
    chunks[0].fill_(21)
    chunks[1].fill_(22)
    if switch_current:
        env.compute = _Stream(env, "post_sfa")
    # Work added AFTER prepare must precede the native write, not overwrite it.
    env.compute.queue.append(lambda: planes[0].fill_(31))
    ticket = env.connector.submit_layerwise_prefill_row(row)
    env.connector.wait_layerwise_prefill_row(ticket)
    observed = []
    env.compute.queue.append(
        lambda: observed.append(
            planes[0].as_subclass(torch.Tensor).view(12, -1).clone()
        )
    )
    assert not observed and env.pending
    env.compute.run_to(len(env.compute.queue))
    assert not env.pending
    assert bool((observed[0][[4, 0]] == 21).all())
    assert bool((observed[0][9] == 22).all())
    assert bool((observed[0][1] == 31).all())
    assert env.connector._layerwise_prefill_inflight  # Device wait did not release.
    env.connector.complete_layerwise_prefill_row(ticket)


@pytest.mark.parametrize("complete_store_first", [False, True])
def test_async_old_bank_event_orders_load_after_store(row_env, complete_store_first):
    env = row_env
    store = _prepare_async(env)
    chunks, src = store._args[:2]
    dst = env.planes(1)
    dst[0].fill_(-9)
    load = env.connector.prepare_layerwise_prefill_row(
        dst,
        chunks,
        [0, 2],
        [2, 3],
        torch.tensor([4, 0, 9]),
        kv_group=1,
        direction=False,
    )
    env.connector.submit_layerwise_prefill_row(store)
    if complete_store_first:
        env.connector.complete_layerwise_prefill_row(store)
    event = store.done_event
    env.connector.submit_layerwise_prefill_row(load, wait_event=event)
    env.connector.wait_layerwise_prefill_row(load)
    # Running just the consumer recursively executes event dependencies, not
    # every stream. Missing wait_event would read the CPU sentinel instead.
    env.compute.run_to(len(env.compute.queue))
    assert not env.pending
    torch.testing.assert_close(
        dst[0].as_subclass(torch.Tensor).view(12, -1)[[4, 0, 9]],
        src[0].as_subclass(torch.Tensor).view(12, -1)[[4, 0, 9]],
    )
    assert ("load", "wait_event", event) in env.events
    env.connector.drain_layerwise_prefill_transfers()
    assert not env.connector._layerwise_prefill_inflight


def test_async_completion_fences_event_not_later_stream_work(row_env):
    env = row_env
    first, later = _prepare_async(env), _prepare_async(env)
    env.connector.submit_layerwise_prefill_row(first)
    env.connector.submit_layerwise_prefill_row(later)
    env.connector.complete_layerwise_prefill_row(first)
    assert len(env.pending) == 1
    assert list(env.connector._layerwise_prefill_inflight.values()) == [later]
    assert not any(e[1] == "sync" for e in env.events)
    env.connector.drain_layerwise_prefill_transfers()
    assert not env.pending


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("prepared_slots", [False, True])
def test_async_metadata_snapshot_and_inference_payloads(
    row_env, inference, prepared_slots
):
    env = row_env
    with torch.inference_mode(inference):
        mapping = torch.tensor([4, 0, 9])
        plan = (
            env.connector.prepare_layerwise_prefill_slots(
                mapping,
                kv_group=1,
                capacity=12,
            )
            if prepared_slots
            else mapping
        )
        row = _prepare_async(env, slots=plan)
        for tensor, version in row._metadata_versions:
            assert not torch.is_inference(tensor) and tensor._version == version
        mapping.fill_(-1)
        # Native config is all values, not a reference to mutable layout mirrors.
        env.connector.k_hidden_dims = -1
        env.connector.kv_format = KVCacheFormat.UNDEFINED
        env.connector.submit_layerwise_prefill_row(row)
        env.connector.complete_layerwise_prefill_row(row)
    assert env.calls[0]["slots"] == [4, 0, 9]
    assert env.calls[0]["widths"] == (32, 0, 32)


@pytest.mark.parametrize(
    "index", [2, 3, 4, 14, "host_ptrs", "host_offsets", "host_sizes"]
)
@pytest.mark.parametrize("inference", [False, True])
def test_async_same_value_metadata_write_is_rejected(row_env, index, inference):
    env = row_env
    row = _prepare_async(env)
    if isinstance(index, str):
        host_index = ("host_ptrs", "host_offsets", "host_sizes").index(index)
        upload = env.uploads[-3 + host_index]
        tensor = upload["src"]()
        assert tensor.is_pinned() and not upload["complete"]
    else:
        tensor = row._args[index]
    with torch.inference_mode(inference):
        tensor[0] = tensor[0]  # Even unchanged/all-ones metadata must fail closed.
    env.events.clear()
    with pytest.raises(ValueError, match="metadata was mutated"):
        env.connector.submit_layerwise_prefill_row(row)
    assert (
        not env.events
        and not env.calls
        and not env.connector._layerwise_prefill_inflight
    )
    env.connector.synchronize_shared_cpu_store_publication()


@pytest.mark.parametrize(
    "case",
    [
        "connector",
        "copy",
        "layout",
        "group",
        "capacity",
        "chunk",
        "storage",
        "dtype",
        "stream",
        "device",
        "format",
        "width",
        "slots",
    ],
)
def test_async_binding_or_tensor_identity_mutation_fails_before_enqueue(row_env, case):
    env = row_env
    plan = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([4, 0, 9]),
        kv_group=1,
        capacity=12,
    )
    row = _prepare_async(env, slots=plan)
    connector = env.connector
    if case == "connector":
        connector = copy.copy(connector)
    elif case == "copy":
        row = copy.copy(row)
    elif case == "layout":
        connector._group_layouts[1] = copy.copy(connector._group_layouts[1])
    elif case == "group":
        connector._group_layouts[1] = connector._group_layouts[0]
    elif case == "capacity":
        row._args[1][0].resize_(2, 4, 1, 32)
    elif case == "chunk":
        row._args[0][0].resize_(1)
    elif case == "storage":
        row._args[0][0].set_(torch.zeros_like(row._args[0][0]))
    elif case == "dtype":
        connector.dtype = torch.float16
    elif case == "stream":
        connector.store_stream = _Stream(env, "other_store")
    elif case == "device":
        connector.store_stream.device = _OTHER_DEVICE
    elif case == "format":
        connector._group_layouts[1].kv_format = KVCacheFormat.MLA_LATENT
    elif case == "width":
        connector._group_layouts[1].dsa_hidden_dims = 64
    elif case == "slots":
        plan._slots.fill_(1)
    env.events.clear()
    with pytest.raises(ValueError):
        connector.submit_layerwise_prefill_row(row)
    assert (
        not env.events
        and not env.calls
        and not env.connector._layerwise_prefill_inflight
    )


def test_async_ticket_lifecycle_and_frozen_fields(row_env):
    env = row_env
    row = _prepare_async(env)
    with pytest.raises(ValueError, match="requires successful submission"):
        _ = row.done_event
    for method in ("wait_layerwise_prefill_row", "complete_layerwise_prefill_row"):
        with pytest.raises(ValueError):
            getattr(env.connector, method)(row)
        with pytest.raises(ValueError, match="another connector"):
            getattr(copy.copy(env.connector), method)(row)
    for name, value in (
        ("_args", ()),
        ("_kv_group", 0),
        ("_state", "complete"),
        ("done_event", None),
        ("_direction", False),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(row, name, value)
    env.connector.submit_layerwise_prefill_row(row)
    with pytest.raises(ValueError, match="already been submitted"):
        env.connector.submit_layerwise_prefill_row(row)
    event = row.done_event
    env.connector.complete_layerwise_prefill_row(row)
    gc.collect()
    assert all(ref() is None for ref in env.last_refs)
    env.connector.complete_layerwise_prefill_row(row)  # Idempotent, no extra fence.
    env.connector.wait_layerwise_prefill_row(row)
    assert row.done_event is event
    with pytest.raises(ValueError, match="already been submitted"):
        env.connector.submit_layerwise_prefill_row(row)
    assert sum(e[1] == "event_sync" for e in env.events) == 1


@pytest.mark.parametrize("failure", ["launch", "record", "producer_record", "wait"])
def test_async_partial_submission_errors_are_registered_and_drained(
    row_env,
    monkeypatch,
    failure,
):
    env = row_env
    good = _prepare_async(env, direction=False)
    bad = _prepare_async(env)
    env.connector.submit_layerwise_prefill_row(good)
    good_refs = env.last_refs
    if failure == "launch":
        env.launch_error = True
    elif failure == "record":
        env.event_record_error = bad._done_event
    elif failure == "producer_record":
        env.event_record_error = bad._producer_event
    else:
        bad_id, bad_ref = id(bad), weakref.ref(bad)

        def fail_wait(_producer):
            assert env.connector._layerwise_prefill_inflight[bad_id] is bad_ref()
            raise RuntimeError("wait failed")

        monkeypatch.setattr(env.store, "wait_event", fail_wait)
    with pytest.raises(RuntimeError, match="failed"):
        env.connector.submit_layerwise_prefill_row(bad)
    assert list(env.connector._layerwise_prefill_inflight.values()) == [good, bad]
    assert not any(e[1] in ("sync", "event_sync") for e in env.events)
    assert bad._recorded is False  # Preparation's stale event is not completion.
    with pytest.raises(ValueError):
        env.connector.wait_layerwise_prefill_row(bad)
    refs = (
        good_refs
        + env.last_refs
        + [
            upload[name]
            for upload in env.uploads
            if upload["non_blocking"]
            for name in ("src", "dst")
        ]
    )
    del good, bad
    gc.collect()
    assert all(ref() is not None for ref in refs)
    with pytest.raises(RuntimeError, match="failed"):
        env.connector.drain_layerwise_prefill_transfers()
    assert not env.pending and not env.connector._layerwise_prefill_inflight
    assert ("store", "sync") in env.events
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_async_complete_partial_launch_uses_event_and_preserves_error(row_env):
    env = row_env
    row = _prepare_async(env)
    env.launch_error = True
    with pytest.raises(RuntimeError, match="native launch failed") as launch:
        env.connector.submit_layerwise_prefill_row(row)
    with pytest.raises(RuntimeError, match="native launch failed") as completion:
        env.connector.complete_layerwise_prefill_row(row)
    assert completion.value is launch.value
    assert not env.pending and not env.connector._layerwise_prefill_inflight
    assert not any(e[1] == "sync" for e in env.events)
    assert row._state == "failed" and not row._owners


@pytest.mark.parametrize("partial", [False, True])
def test_async_unknown_fence_poison_quarantines_owners_and_drain_retries(
    row_env, partial
):
    env = row_env
    row = _prepare_async(env)
    upload_refs = [u[name] for u in env.uploads[-3:] for name in ("src", "dst")]
    env.launch_error = partial
    if partial:
        with pytest.raises(RuntimeError, match="native launch failed"):
            env.connector.submit_layerwise_prefill_row(row)
    else:
        env.connector.submit_layerwise_prefill_row(row)
    env.sync_error = True
    with pytest.raises(RuntimeError, match="native launch failed|completion failed"):
        env.connector.drain_layerwise_prefill_transfers()
    assert row._owners and env.pending
    assert env.connector._layerwise_prefill_inflight[id(row)] is row
    gc.collect()
    assert all(ref() is not None for ref in env.last_refs + upload_refs)
    for action in (
        lambda: _prepare_async(env),
        lambda: env.connector.submit_layerwise_prefill_row(row),
        lambda: env.connector.transfer_layerwise_prefill_row(
            (),
            (),
            (),
            (),
            None,
            kv_group=1,
            direction=True,
        ),
    ):
        with pytest.raises(RuntimeError, match="previous prefill row completion fence"):
            action()
    env.sync_error = False
    with pytest.raises(RuntimeError, match="native launch failed|previous prefill row"):
        env.connector.drain_layerwise_prefill_transfers()
    assert not row._owners and not env.pending
    assert not env.connector._layerwise_prefill_inflight
    gc.collect()
    assert all(ref() is None for ref in env.last_refs + upload_refs)


def test_async_drain_attempts_remaining_rows_after_failed_event_fence(
    row_env, monkeypatch
):
    env = row_env
    first, second = _prepare_async(env), _prepare_async(env, direction=False)
    env.connector.submit_layerwise_prefill_row(first)
    env.connector.submit_layerwise_prefill_row(second)

    def fail():
        raise RuntimeError("first fence failed")

    with monkeypatch.context() as patch:
        patch.setattr(first.done_event, "synchronize", fail)
        with pytest.raises(RuntimeError, match="first fence failed"):
            env.connector.drain_layerwise_prefill_transfers()
        assert list(env.connector._layerwise_prefill_inflight.values()) == [first]
        assert second._state == "complete" and not second._owners
    with pytest.raises(RuntimeError, match="previous prefill row"):
        env.connector.drain_layerwise_prefill_transfers()


def test_async_empty_restore_has_dependency_event_without_native(row_env):
    env = row_env
    env.connector._group_layouts.clear()
    row = env.connector.prepare_layerwise_prefill_row(
        (),
        (),
        (),
        (),
        None,
        kv_group=1,
        direction=False,
    )
    assert not row._args
    env.connector.submit_layerwise_prefill_row(row)
    env.connector.wait_layerwise_prefill_row(row)
    env.connector.complete_layerwise_prefill_row(row)
    assert row._state == "complete" and not env.calls
    assert ("compute", "wait_event", row.done_event) in env.events
    assert not any(e[1] == "sync" for e in env.events)


def test_async_hooks_reject_storage_executor_thread(row_env, monkeypatch):
    env = row_env
    row = _prepare_async(env)
    monkeypatch.setattr(npu_connectors, "get_ident", lambda: -1)
    for action in (
        lambda: _prepare_async(env),
        lambda: env.connector.submit_layerwise_prefill_row(row),
        lambda: env.connector.wait_layerwise_prefill_row(row),
        lambda: env.connector.complete_layerwise_prefill_row(row),
        env.connector.drain_layerwise_prefill_transfers,
    ):
        with pytest.raises(RuntimeError, match="model/control thread"):
            action()
    assert not env.calls


@pytest.mark.parametrize(
    "failure",
    [
        "registration",
        "allocation",
        "host_allocation",
        "copy_before",
        "copy_after",
        "event",
    ],
)
@pytest.mark.parametrize("fence_fails", [False, True])
def test_async_preparation_failure_fences_metadata_or_quarantines(
    row_env,
    monkeypatch,
    failure,
    fence_fails,
):
    env = row_env
    env.sync_error = fence_fails
    if failure == "registration":
        env.pointer_error = "raise"
    elif failure == "event":
        env.event_record_error = True
    elif failure.startswith("copy_"):
        prepare = env.connector._prepare_layerwise_prefill_row_metadata
        tensor_copy = torch.Tensor.copy_

        def preparing(validated, direction, owners, **kwargs):
            env.preparing_owners = owners
            try:
                return prepare(validated, direction, owners, **kwargs)
            finally:
                env.preparing_owners = None

        def fail_copy(dst, src, non_blocking=False):
            assert non_blocking and src.is_pinned()
            assert any(owner is src for owner in env.preparing_owners)
            assert any(owner is dst for owner in env.preparing_owners)
            if dst.dtype == torch.int32 and failure == "copy_before":
                raise RuntimeError("metadata copy failed before enqueue")
            result = tensor_copy(dst, src, non_blocking=non_blocking)
            if dst.dtype == torch.int32:
                raise RuntimeError("metadata copy failed after enqueue")
            return result

        monkeypatch.setattr(
            env.connector, "_prepare_layerwise_prefill_row_metadata", preparing
        )
        monkeypatch.setattr(torch.Tensor, "copy_", fail_copy)
    else:
        name = "tensor" if failure == "host_allocation" else "empty"
        factory = getattr(torch, name)

        def fail_sizes(*args, **kwargs):
            if kwargs.get("dtype") == torch.int32 and (
                kwargs.get("device") is _DEVICE or kwargs.get("pin_memory")
            ):
                raise RuntimeError("metadata allocation failed")
            return factory(*args, **kwargs)

        monkeypatch.setattr(torch, name, fail_sizes)
    with pytest.raises(RuntimeError, match="failed"):
        _prepare_async(env)
    assert not env.calls and not env.connector._layerwise_prefill_inflight
    assert env.events[-1] == ("store", "sync")
    if failure != "registration":
        assert any(upload["non_blocking"] for upload in env.uploads)
    if failure == "copy_after":
        assert sum(upload["non_blocking"] for upload in env.uploads) == 2
    if fence_fails:
        assert env.connector._layerwise_prefill_row_failed_owners
        refs = [
            weakref.ref(owner)
            for owner in env.connector._layerwise_prefill_row_failed_owners
            if isinstance(owner, (torch.Tensor, _Event))
        ]
        env.events.clear()  # Event tracing must not mask lost connector ownership.
        gc.collect()
        assert all(ref() is not None for ref in refs)
        with pytest.raises(RuntimeError, match="completion failed"):
            env.connector.drain_layerwise_prefill_transfers()
        assert all(ref() is not None for ref in refs)
        env.sync_error = False
        with pytest.raises(RuntimeError, match="previous prefill row"):
            env.connector.drain_layerwise_prefill_transfers()
        gc.collect()
        assert all(ref() is None for ref in refs)
    else:
        assert (
            getattr(env.connector, "_layerwise_prefill_row_failed_owners", None) is None
        )
    assert all(upload["complete"] for upload in env.uploads)


@pytest.mark.parametrize("recognition", ["false", "missing", "error"])
def test_async_metadata_requires_recognized_pinned_allocator(
    row_env, monkeypatch, recognition
):
    env = row_env
    slots = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([4, 0, 9]), kv_group=1, capacity=12
    )
    before = len(env.uploads)

    def is_pinned(_tensor):
        if recognition == "error":
            raise RuntimeError("pinned recognition unavailable")
        return False

    monkeypatch.setattr(
        torch.Tensor, "is_pinned", None if recognition == "missing" else is_pinned
    )
    with pytest.raises(RuntimeError, match="pinned"):
        _prepare_async(env, slots=slots)
    assert len(env.uploads) == before and not env.calls
    assert env.events[-1] == ("store", "sync")


@pytest.mark.parametrize("fence_fails", [False, True])
def test_async_unsubmitted_metadata_owners_survive_existing_abort_fences(
    row_env, fence_fails
):
    env = row_env
    slots = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor([4, 0, 9]), kv_group=1, capacity=12
    )
    rows = [_prepare_async(env, direction=d, slots=slots) for d in (False, True)]
    uploads = env.uploads[1:]
    refs = [upload[name] for upload in uploads for name in ("src", "dst")]
    env.connector.drain_layerwise_prefill_transfers()
    assert not env.connector._layerwise_prefill_inflight
    assert all(not upload["complete"] for upload in uploads)
    # Backend abort retains prepared tickets and fences BOTH existing streams,
    # including work with no KV submission/completion event in the registry.
    env.sync_error = fence_fails
    for fence in (
        env.connector.synchronize_dense_load_stream,
        env.connector.synchronize_shared_cpu_store_publication,
    ):
        if fence_fails:
            with pytest.raises(RuntimeError, match="completion failed"):
                fence()
        else:
            fence()
    gc.collect()
    assert all(ref() is not None for ref in refs)
    if fence_fails:
        assert all(not upload["complete"] for upload in uploads)
        env.sync_error = False
        env.connector.synchronize_dense_load_stream()
        env.connector.synchronize_shared_cpu_store_publication()
    assert all(upload["complete"] for upload in uploads) and not env.calls
    del rows
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_async_101_rows_four_banks_upload_only_pinned_chunk_metadata(row_env):
    env = row_env
    slots = {
        (group, bank): env.connector.prepare_layerwise_prefill_slots(
            torch.tensor([4, 0, 9]), kv_group=group, capacity=12
        )
        for group in (0, 1)
        for bank in (0, 1)
    }
    before = env.host_ops.copy()
    for group, count in ((0, 79), (1, 22)):
        for _ in range(count):
            for bank in (0, 1):
                row = _prepare_async(
                    env, group=group, direction=not bank, slots=slots[group, bank]
                )
                env.connector.submit_layerwise_prefill_row(row)
    del row
    assert len(env.pending) == 202
    assert env.host_ops == {
        **before,
        "uploads": before["uploads"] + 606,
        "slices": before["slices"] + 202,
    }
    uploads = env.uploads[4:]
    assert len(uploads) == 606
    assert all(
        upload["non_blocking"] and upload["pinned"] and not upload["complete"]
        for upload in uploads
    )
    assert sum(upload["nbytes"] for upload in uploads) == 202 * 2 * 16
    assert not any(
        e[1] in ("sync", "event_sync", "copy_sync")
        for e in env.events
        if e[0] != "compute"
    )
    env.connector.drain_layerwise_prefill_transfers()
    gc.collect()
    assert all(upload["complete"] for upload in uploads)
    assert all(upload[name]() is None for upload in uploads for name in ("src", "dst"))


def test_async_partial_launch_and_failed_complete_preserves_both_errors(row_env):
    env = row_env
    row = _prepare_async(env)
    env.launch_error = True
    with pytest.raises(RuntimeError, match="native launch failed"):
        env.connector.submit_layerwise_prefill_row(row)
    env.event_record_error = True
    with pytest.raises(RuntimeError, match="native launch failed") as error:
        env.connector.complete_layerwise_prefill_row(row)
    assert str(error.value.__cause__) == "event record failed"
    assert row._owners and env.connector._layerwise_prefill_inflight
    assert not any(e[1] in ("sync", "event_sync") for e in env.events)
    # drain must not retry the broken event: it uses the recorded transfer stream.
    with pytest.raises(RuntimeError, match="native launch failed"):
        env.connector.drain_layerwise_prefill_transfers()
    assert not env.pending and not row._owners


def test_async_512_chunk_metadata_is_row_owned_and_never_refilled(row_env):
    env = row_env
    sizes = [1] * 511 + [3]
    planes = env.planes(1, blocks=129)
    slots = env.connector.prepare_layerwise_prefill_slots(
        torch.arange(514),
        kv_group=1,
        capacity=516,
    )
    first = env.connector.prepare_layerwise_prefill_row(
        planes,
        _chunks(planes, sizes),
        list(range(512)),
        list(range(1, 512)) + [514],
        slots,
        kv_group=1,
        direction=True,
    )
    small = _prepare_async(env)
    assert first._args[4].data_ptr() != small._args[4].data_ptr()
    snapshots = tuple(
        (t, version, t.clone())
        for t, version in first._metadata_versions
        if t.device.type == "cpu"
    )
    assert len(snapshots) == 3
    assert sum(t.numel() * t.element_size() for t, _, _ in snapshots) == 512 * 16
    before = env.host_ops.copy()
    env.connector.submit_layerwise_prefill_row(first)
    env.connector.submit_layerwise_prefill_row(small)
    assert env.host_ops == before
    assert "sizes" not in env.calls[0]
    env.store.run_to(len(env.store.queue))
    assert env.calls[0]["sizes"] == sizes
    assert env.calls[0]["offsets"] == list(range(512))
    for tensor, version, snapshot in snapshots:
        assert tensor._version == version
        torch.testing.assert_close(tensor, snapshot)
    for upload in env.uploads[1:4]:
        torch.testing.assert_close(
            upload["dst"]().as_subclass(torch.Tensor), upload["src"]()
        )
    env.connector.drain_layerwise_prefill_transfers()
    assert not env.pending
