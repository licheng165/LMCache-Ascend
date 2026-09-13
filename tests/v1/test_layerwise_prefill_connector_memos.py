# SPDX-License-Identifier: Apache-2.0
"""Connector-side prefix memos: chunk geometry/pointer reuse and slot validation.

The chunk memo (weakref id + expected packed numel + resolved device pointer)
and the identity-keyed incremental slot-validation cache must be observably
equivalent to fresh full validation: same accept/reject decisions, same native
pointer values, no reuse across dead objects, and failures never memoized.
"""

# Standard
import gc

# Third Party
import pytest
import torch

# Local
from tests.v1.test_layerwise_prefill_row_transfer import (  # noqa: F401
    _chunks,
    row_env,
)


@pytest.fixture
def memos_row_env(request: pytest.FixtureRequest):
    return request.getfixturevalue("row_env")


def _registrations(env) -> int:
    return sum(event[1] == "registered" for event in env.events)


def _slot_values(plan) -> list:
    return plan._slots.as_subclass(torch.Tensor).tolist()


def test_chunk_memo_reuses_pointer_resolution_across_calls(memos_row_env):
    env = memos_row_env
    planes = env.planes(0)
    chunks = _chunks(planes, [2, 3, 1])
    starts, ends = [0, 2, 5], [2, 5, 6]
    mapping = torch.arange(6, dtype=torch.int64)
    call = lambda: env.connector.transfer_layerwise_prefill_row(
        planes, chunks, starts, ends, mapping, kv_group=0, direction=True
    )
    call()
    assert _registrations(env) == 3
    assert len(env.calls) == 1
    env.events.clear()
    env.calls.clear()
    call()
    assert _registrations(env) == 0
    assert env.calls[0]["chunk_ptrs"] == [chunk.data_ptr() for chunk in chunks]
    memo = env.connector._layerwise_prefill_chunk_memo
    assert [memo[id(chunk)][2] for chunk in chunks] == [
        chunk.data_ptr() + 4096 for chunk in chunks
    ]
    extra = _chunks(planes, [4])
    env.events.clear()
    env.connector.transfer_layerwise_prefill_row(
        planes,
        [*chunks, *extra],
        [*starts, 6],
        [*ends, 10],
        torch.arange(10, dtype=torch.int64),
        kv_group=0,
        direction=True,
    )
    assert _registrations(env) == 1


def test_chunk_geometry_failures_are_never_memoized(memos_row_env):
    env = memos_row_env
    planes = env.planes(0)
    good = _chunks(planes, [2])
    mapping = torch.arange(2, dtype=torch.int64)
    env.connector.transfer_layerwise_prefill_row(
        planes, good, [0], [2], mapping, kv_group=0, direction=True
    )
    bad = torch.full((7,), -1, dtype=torch.bfloat16)
    for _ in range(2):
        with pytest.raises(ValueError, match="packed plane sizes"):
            env.connector.transfer_layerwise_prefill_row(
                planes, [bad], [0], [2], mapping, kv_group=0, direction=True
            )
    env.connector.transfer_layerwise_prefill_row(
        planes, good, [0], [2], mapping, kv_group=0, direction=True
    )
    assert len(env.calls) == 2


def test_chunk_memo_does_not_share_recycled_objects(memos_row_env):
    env = memos_row_env
    planes = env.planes(1)
    width = sum(plane.shape[-2] * plane.shape[-1] for plane in planes)
    chunks = [torch.arange(2 * width, dtype=torch.bfloat16).view(-1)]
    mapping = torch.arange(2, dtype=torch.int64)
    env.connector.transfer_layerwise_prefill_row(
        planes, chunks, [0], [2], mapping, kv_group=1, direction=True
    )
    memo = env.connector._layerwise_prefill_chunk_memo
    assert len(memo) == 1
    del chunks
    gc.collect()
    assert all(entry[0]() is None for entry in memo.values())
    fresh = [torch.arange(2 * width, dtype=torch.bfloat16).view(-1)]
    env.events.clear()
    env.connector.transfer_layerwise_prefill_row(
        planes, fresh, [0], [2], mapping, kv_group=1, direction=True
    )
    assert _registrations(env) == 1


def test_slots_incremental_matches_full_validation(memos_row_env):
    env = memos_row_env
    capacity = 512
    base = torch.randperm(capacity)[:192]
    identity = ("req", 1, 0, 0)
    lengths = (64, 128, 176)
    plans = []
    for length in lengths:
        mapping = base[:length].to(torch.int64)
        plans.append(
            env.connector.prepare_layerwise_prefill_slots(
                mapping, kv_group=0, capacity=capacity, identity=identity
            )
        )
        reference = env.connector.prepare_layerwise_prefill_slots(
            mapping, kv_group=0, capacity=capacity
        )
        assert _slot_values(plans[-1]) == _slot_values(reference)
    cache = env.connector._layerwise_prefill_slots_cache
    assert list(cache) == [identity]
    assert cache[identity][2] == lengths[-1]
    assert cache[identity][0] == 0 and cache[identity][1] == capacity


def test_slots_incremental_rejects_bad_tails(memos_row_env):
    env = memos_row_env
    capacity = 512
    base = torch.randperm(capacity)[:200]
    identity = ("req", 1, 1, 0)
    env.connector.prepare_layerwise_prefill_slots(
        base[:64].to(torch.int64), kv_group=1, capacity=capacity, identity=identity
    )
    duplicate_tail = torch.tensor(
        base[:64].tolist() + [base[100].item()] * 2, dtype=torch.int64
    )
    with pytest.raises(ValueError, match="duplicate slots"):
        env.connector.prepare_layerwise_prefill_slots(
            duplicate_tail, kv_group=1, capacity=capacity, identity=identity
        )
    prefix_copy_tail = torch.tensor(
        base[:64].tolist() + [base[0].item()], dtype=torch.int64
    )
    with pytest.raises(ValueError, match="duplicate slots"):
        env.connector.prepare_layerwise_prefill_slots(
            prefix_copy_tail, kv_group=1, capacity=capacity, identity=identity
        )
    out_of_range = torch.tensor(base[:64].tolist() + [capacity], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of KV capacity"):
        env.connector.prepare_layerwise_prefill_slots(
            out_of_range, kv_group=1, capacity=capacity, identity=identity
        )
    good = env.connector.prepare_layerwise_prefill_slots(
        base[:80].to(torch.int64), kv_group=1, capacity=capacity, identity=identity
    )
    assert sorted(_slot_values(good)) == sorted(base[:80].tolist())


def test_slots_prefix_change_falls_back_and_replaces_entry(memos_row_env):
    env = memos_row_env
    capacity = 256
    identity = ("req", 2, 0, 1)
    first = torch.randperm(capacity)[:64]
    env.connector.prepare_layerwise_prefill_slots(
        first.to(torch.int64), kv_group=0, capacity=capacity, identity=identity
    )
    second = torch.randperm(capacity)[:96]
    plan = env.connector.prepare_layerwise_prefill_slots(
        second.to(torch.int64), kv_group=0, capacity=capacity, identity=identity
    )
    assert sorted(_slot_values(plan)) == sorted(second.tolist())
    cache = env.connector._layerwise_prefill_slots_cache
    assert cache[identity][2] == 96
    absent = next(value for value in range(capacity) if value not in second.tolist())
    extended = env.connector.prepare_layerwise_prefill_slots(
        torch.tensor(second.tolist() + [absent], dtype=torch.int64),
        kv_group=0,
        capacity=capacity,
        identity=identity,
    )
    assert extended._length == 97
    mismatch = env.connector.prepare_layerwise_prefill_slots(
        first.to(torch.int64), kv_group=0, capacity=capacity, identity=identity
    )
    assert sorted(_slot_values(mismatch)) == sorted(first.tolist())


def test_slots_shorter_mapping_revalidates_fully(memos_row_env):
    env = memos_row_env
    capacity = 256
    identity = ("req", 1, 0, 0)
    values = torch.randperm(capacity)[:128]
    env.connector.prepare_layerwise_prefill_slots(
        values.to(torch.int64), kv_group=0, capacity=capacity, identity=identity
    )
    plan = env.connector.prepare_layerwise_prefill_slots(
        values[:32].to(torch.int64),
        kv_group=0,
        capacity=capacity,
        identity=identity,
    )
    assert _slot_values(plan) == values[:32].tolist()
    assert env.connector._layerwise_prefill_slots_cache[identity][2] == 32


def test_chunk_memo_purge_is_amortized_not_per_insert(memos_row_env):
    env = memos_row_env
    connector = env.connector
    # Regression: a fixed purge threshold rescans the whole memo on EVERY
    # insert once a live-prefix workload (entries never die) crosses it -
    # 0912-2 deployed as O(N)-per-insert and exploded prepare_load/save.
    # Hysteresis must instead sweep once and double past the live size.
    connector._layerwise_prefill_chunk_memo_limit = 16
    planes = env.planes(0)
    mapping = torch.tensor([0], dtype=torch.int64)
    batches = 12
    per_batch = 9
    retained = []
    for _ in range(batches):
        chunks = _chunks(planes, [1] * per_batch)
        retained.extend(chunks)
        for chunk in chunks:
            env.connector.transfer_layerwise_prefill_row(
                planes, [chunk], [0], [1], mapping, kv_group=0, direction=True
            )
    memo = connector._layerwise_prefill_chunk_memo
    assert len(memo) == batches * per_batch
    assert connector._layerwise_prefill_chunk_memo_limit >= 2 * len(memo)
    assert _registrations(env) == batches * per_batch
    assert all(entry[0]() is not None for entry in memo.values())
    # A fresh wave of live inserts must not shrink the limit below the
    # doubled live size (no rescan thrash): the limit only grows.
    before = connector._layerwise_prefill_chunk_memo_limit
    for chunk in _chunks(planes, [1] * 4):
        env.connector.transfer_layerwise_prefill_row(
            planes, [chunk], [0], [1], mapping, kv_group=0, direction=True
        )
    assert connector._layerwise_prefill_chunk_memo_limit >= before
    assert len(memo) == batches * per_batch + 4
    assert all(retained[batch * per_batch] is not None for batch in range(batches))


def test_chunk_memo_dead_entries_are_reclaimed_by_hysteresis(memos_row_env):
    env = memos_row_env
    connector = env.connector
    connector._layerwise_prefill_chunk_memo_limit = 8
    planes = env.planes(0)
    mapping = torch.tensor([0], dtype=torch.int64)

    def wave(count: int) -> list:
        chunks = _chunks(planes, [1] * count)
        for chunk in chunks:
            env.connector.transfer_layerwise_prefill_row(
                planes, [chunk], [0], [1], mapping, kv_group=0, direction=True
            )
        return chunks

    import gc

    wave(6)
    gc.collect()
    live_chunks = wave(6)
    memo = connector._layerwise_prefill_chunk_memo
    # The second wave's sweep (triggered past the limit) must have dropped
    # the first wave's dead entries instead of growing unboundedly.
    assert len(memo) <= 12
    live = sum(1 for entry in memo.values() if entry[0]() is not None)
    assert live == 6
    assert memo[id(live_chunks[0])][0]() is live_chunks[0]


def test_slots_cache_is_bounded_fifo(memos_row_env):
    env = memos_row_env
    capacity = 128
    for index in range(33):
        env.connector.prepare_layerwise_prefill_slots(
            torch.arange(8, dtype=torch.int64),
            kv_group=0,
            capacity=capacity,
            identity=("req", index, 0, 0),
        )
    cache = env.connector._layerwise_prefill_slots_cache
    assert len(cache) == 32
    assert ("req", 0, 0, 0) not in cache
    assert ("req", 32, 0, 0) in cache
