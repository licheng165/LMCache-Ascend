# SPDX-License-Identifier: Apache-2.0
"""Exact row-key projections without reconstructing retained full chunks."""

# ruff: noqa: F811

# Standard
from collections import Counter
from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

# Third Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
import pytest
import torch

# First Party
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillSyncBackend,
    _RowPrefix,
)

# Local
from tests.v1.test_layerwise_prefill_async import _backend, _execute
from tests.v1.test_layerwise_prefill_sync import (
    _metadata,
    _registry,
    _request,
    _step,
    _view,
    runtime,  # noqa: F401
)
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401


@pytest.fixture
def constructions(monkeypatch: pytest.MonkeyPatch) -> Counter:
    counts = Counter()
    project = CacheEngineKey.get_layer

    def allocate(key: CacheEngineKey, row: int) -> LayerCacheEngineKey:
        counts["keys"] += 1
        return project(key, row)

    # Patching inherited __new__ can leave CPython's type slot changed after
    # fixture teardown. Count the one-key projection without changing allocation.
    monkeypatch.setattr(CacheEngineKey, "get_layer", allocate)
    return counts


def _install_plan(backend: Any, req: Any, end: int) -> None:
    for group in (0, 1):
        backend._plans[req.request_id, group, end] = [
            (
                start,
                min(start + 256, end),
                CacheEngineKey(
                    "plan-model",
                    8,
                    0,
                    req.token_ids[0] + start + min(start + 256, end),
                    torch.bfloat16,
                    deepcopy(req.request_configs),
                    kv_group=group,
                ),
            )
            for start in range(0, end, 256)
        ]


def test_first_32_steps_construct_only_51712_new_row_keys(
    constructions: Counter,
) -> None:
    backend = object.__new__(LayerwisePrefillSyncBackend)
    backend._prefixes, backend._plans = {}, {}
    req = replace(
        _request(),
        token_ids=tuple(range(32 * 4096)),
        block_ids_by_bank=((tuple(range(1, 1025)),) * 2,) * 2,
    )
    for step in range(32):
        req = replace(
            req,
            compute_start=step * 4096,
            restore_end=step * 4096,
            compute_end=(step + 1) * 4096,
        )
        _install_plan(backend, req, req.compute_end)
        before = constructions["keys"]
        for group, count in enumerate((79, 22)):
            for row in range(count):
                identity = (req.request_id, 1, group, row)
                prior = backend._prefixes.get(identity)
                starts, ends, keys = backend._plan(req, req.compute_end, group, row)
                assert starts == list(range(0, req.compute_end, 256))
                assert ends == list(range(256, req.compute_end + 1, 256))
                if prior is not None:
                    assert all(a is b for a, b in zip(keys, prior.keys, strict=False))
                assert all(
                    key.layer_id == row and key.kv_group == group for key in keys
                )
                backend._prefixes[identity] = _RowPrefix(starts, ends, keys, [], 1, 11)
        # The cold step builds its full plan; every warm step builds 16 per row.
        assert constructions["keys"] - before == 16 * 101
        backend._plans.clear()
    assert constructions["keys"] == 51712
    assert sum(range(1, 33)) * 16 * 101 == 853248


@pytest.mark.parametrize(
    "start,end,restore",
    [(0, 530, 0), (300, 530, 300), (299, 300, 300), (255, 256, 256), (511, 512, 512)],
)
def test_cold_restore_and_overlap_construct_the_full_changed_suffix(
    constructions: Counter, start: int, end: int, restore: int
) -> None:
    backend = object.__new__(LayerwisePrefillSyncBackend)
    backend._prefixes, backend._plans = {}, {}
    req = _request(start=start, end=end, restore_end=restore)
    _install_plan(backend, req, restore)
    starts, ends, keys = backend._plan(req, restore, 0, 0)
    assert constructions["keys"] == (restore + 255) // 256
    backend._prefixes[req.request_id, 1, 0, 0] = _RowPrefix(
        starts, ends, keys, [], 1, 11
    )
    _install_plan(backend, req, end)
    before = constructions["keys"]
    _, _, current = backend._plan(req, end, 0, 0)
    keep = min(start, restore) // 256
    assert constructions["keys"] - before == (end + 255) // 256 - keep
    assert all(a is b for a, b in zip(current[:keep], keys[:keep], strict=True))
    assert all(a is not b for a, b in zip(current[keep:], keys[keep:], strict=False))
    if restore == end:
        assert current == keys  # Equal values still require recomputing the overlap.


@pytest.fixture
def planned() -> tuple:
    backend = object.__new__(LayerwisePrefillSyncBackend)
    backend._prefixes, backend._plans = {}, {}
    req = replace(
        _request(start=512, end=530),
        request_configs={"lmcache.tag.tenant": "sentinel", "options": {"ids": [1]}},
    )
    _install_plan(backend, req, req.restore_end)
    starts, ends, keys = backend._plan(req, req.restore_end, 0, 0)
    prior = _RowPrefix(starts, ends, keys, [], 1, 11)
    backend._prefixes[req.request_id, 1, 0, 0] = prior
    _install_plan(backend, req, req.compute_end)
    return backend, req, prior


@pytest.mark.parametrize("side", ["plan", "prior"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "other"),
        ("world_size", 4),
        ("worker_id", 1),
        ("chunk_hash", 999),
        ("dtype", torch.float16),
        (
            "request_configs",
            {"lmcache.tag.tenant": "sentinel", "options": {"ids": [2]}},
        ),
        ("tags", (("tenant", "other"),)),
        ("_dtype_str", "float16"),
        ("kv_group", 1),
        ("start", 1),
        ("end", 255),
    ],
)
def test_reuse_requires_every_projected_field_and_range(
    planned: tuple, constructions: Counter, side: str, field: str, value: Any
) -> None:
    backend, req, prior = planned
    plan = backend._plans[req.request_id, 0, req.compute_end]
    if field in ("start", "end"):
        if side == "prior":
            (prior.starts if field == "start" else prior.ends)[0] = value
        else:
            start, end, key = plan[0]
            plan[0] = (value, end, key) if field == "start" else (start, value, key)
    else:
        key = prior.keys[0] if side == "prior" else plan[0][2]
        if field == "request_configs":
            # In-place nested metadata changes must not be hidden by key equality.
            key.request_configs["options"]["ids"][:] = value["options"]["ids"]
        else:
            setattr(key, field, value)
    before = constructions["keys"]
    starts, ends, keys = backend._plan(req, req.compute_end, 0, 0)
    assert constructions["keys"] - before == 2
    assert keys[0] is not prior.keys[0] and keys[1] is prior.keys[1]
    assert starts == [entry[0] for entry in plan]
    assert ends == [entry[1] for entry in plan]
    # Compare ALL fields, not CacheEngineKey.__eq__'s narrower cache identity.
    for actual, (_, _, key) in zip(keys, plan, strict=True):
        expected = key.get_layer(0)
        assert type(actual) is type(expected)
        assert all(
            getattr(actual, field.name) == getattr(expected, field.name)
            for field in fields(LayerCacheEngineKey)
        )


@pytest.mark.parametrize("identity", ["request", "generation", "group", "row"])
def test_no_reuse_from_another_identity(
    planned: tuple, constructions: Counter, identity: str
) -> None:
    backend, req, prior = planned
    group = row = 0
    if identity == "request":
        req = replace(req, request_id="other")
    elif identity == "generation":
        req = replace(req, allocation_generation=2)
    elif identity == "group":
        group = 1
    else:
        row = 1
    _install_plan(backend, req, req.compute_end)
    before = constructions["keys"]
    _, _, keys = backend._plan(req, req.compute_end, group, row)
    assert constructions["keys"] - before == 3
    assert all(key is not old for key, old in zip(keys, prior.keys, strict=False))
    assert all(key.layer_id == row and key.kv_group == group for key in keys)


@pytest.mark.parametrize("kind", ["base_subclass", "layer_subclass", "layer", "fake"])
def test_nonconcrete_keys_use_original_get_layer_and_equality(
    planned: tuple, constructions: Counter, kind: str
) -> None:
    backend, req, prior = planned
    plan = backend._plans[req.request_id, 0, req.compute_end]
    calls = []

    class CustomKey(CacheEngineKey):
        def get_layer(self, row: int) -> LayerCacheEngineKey:
            calls.append(row)
            return super().get_layer(row + 1)

    class CustomLayerKey(LayerCacheEngineKey):
        pass

    class FakeKey:
        def get_layer(self, row: int) -> tuple:
            calls.append(row)
            return ("fake", row)

    key = plan[0][2]
    if kind == "base_subclass":
        key = CustomKey(**{f.name: getattr(key, f.name) for f in fields(key) if f.init})
    elif kind == "layer_subclass":
        old = prior.keys[0]
        prior.keys[0] = CustomLayerKey(
            **{f.name: getattr(old, f.name) for f in fields(old) if f.init}
        )
    elif kind == "layer":
        key = key.get_layer(99)
    else:
        key = FakeKey()
        prior.keys[0] = ("fake", 0)
    plan[0] = (*plan[0][:2], key)
    before = constructions["keys"]
    _, _, keys = backend._plan(req, req.compute_end, 0, 0)
    assert constructions["keys"] - before == (1 if kind == "fake" else 2)
    assert keys[0] is not prior.keys[0] and keys[1] is prior.keys[1]
    assert calls == ([0] if kind in ("fake", "base_subclass") else [])
    assert (keys[:2] == prior.keys[:2]) is (kind in ("fake", "layer"))


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("request_count", [1, 2])
def test_real_saves_preserve_exact_acks_transfers_and_partial_overlap(
    page_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    constructions: Counter,
    mode: str,
    generation: int,
    request_count: int,
) -> None:
    def run(reuse: bool) -> tuple:
        engine = page_runtime.engine()
        backend = (
            _backend(engine)
            if mode == "async"
            else engine.layerwise_prefill_window_backend
        )
        original = backend._plan
        acknowledge = engine.layerwise_prefill_ack
        acks, counts = [], Counter()

        def plan(req: Any, end: int, group: int, row: int) -> tuple:
            before = constructions["keys"]
            prior = backend._prefixes.get(
                (req.request_id, req.allocation_generation, group, row)
            )
            if reuse:
                result = original(req, end, group, row)
            else:
                entries = backend._plans[req.request_id, group, end]
                result = (
                    [s for s, _, _ in entries],
                    [e for _, e, _ in entries],
                    [key.get_layer(row) for _, _, key in entries],
                )
            counts[req.compute_start] += constructions["keys"] - before
            if reuse and prior is not None:
                keep = (
                    min(req.compute_start, prior.ends[-1] if prior.ends else 0) // 256
                )
                assert all(
                    a is b
                    for a, b in zip(result[2][:keep], prior.keys[:keep], strict=True)
                )
                assert all(
                    a is not b
                    for a, b in zip(result[2][keep:], prior.keys[keep:], strict=False)
                )
            return result

        def ack(identity: Any, error: Any = None) -> None:
            acks.append(deepcopy(identity))
            acknowledge(identity, error)

        monkeypatch.setattr(backend, "_plan", plan)
        monkeypatch.setattr(engine, "layerwise_prefill_ack", ack)
        requests = [_request(i) for i in range(request_count)]
        for start, end in ((0, 300), (300, 530), (530, 560)):
            requests = [
                replace(
                    req,
                    compute_start=start,
                    restore_end=start,
                    compute_end=end,
                    allocation_generation=generation if start else 1,
                    block_ids_by_bank=tuple(
                        tuple(
                            tuple(
                                b + (24 if start == 300 and generation == 2 else 0)
                                for b in blocks
                            )
                            for blocks in groups
                        )
                        for groups in req.block_ids_by_bank
                    ),
                )
                for req in requests
            ]
            (_execute if mode == "async" else _step)(engine, backend, requests)
        assert counts == Counter(
            {
                0: 202 * request_count,
                300: (202 if reuse else 303) * request_count,
                530: (101 if reuse else 303) * request_count,
            }
        )
        transfers = engine.gpu_connector.calls
        assert len(transfers) == 505 * request_count
        assert (
            sum(
                sum(e - s for s, e in zip(starts, ends, strict=True))
                * (4 if group == 0 else 2)
                for _, group, starts, ends in transfers
            )
            == 1452 * (79 * 4 + 22 * 2) * request_count
        )
        for req in requests:
            backend.abort_request(req.request_id)
        return acks, transfers

    assert run(False) == run(True)


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("mutation", ["chunk_hash", "tags", "layer_id", "kv_group"])
def test_warm_save_still_rejects_changed_retained_keys(
    runtime: Any, generation: int, mutation: str
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    _step(engine, backend, [_request()])
    req = _request(start=300, end=530, generation=generation)
    backend.bind_step([req], _registry(engine))
    metadata = _metadata(_view(), _view().rows_by_group[0][0], [req])
    backend.wait_for_load(metadata)
    identity = (req.request_id, generation, 0, 0)
    prior = backend._prefixes[identity]
    changed = replace(prior, keys=deepcopy(prior.keys))
    key = changed.keys[0]
    setattr(
        key,
        mutation,
        (("tenant", "other"),) if mutation == "tags" else getattr(key, mutation) + 1,
    )
    backend._prefixes[identity] = changed
    calls = list(engine.gpu_connector.calls)
    with pytest.raises(ValueError, match="unchanged chunk keys differ"):
        backend.sync_save(metadata, _registry(engine)[metadata.row.layer_name])
    assert engine.gpu_connector.calls == calls
    assert backend._prefixes[identity] is changed
    assert backend._failed


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("mutation", ["configs", "tokens", "start", "end"])
def test_warm_bind_still_rejects_mutable_configs_tokens_and_inexact_chunks(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, generation: int, mutation: str
) -> None:
    engine = runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    req = replace(_request(), request_configs={"options": {"ids": [1]}})
    _step(engine, backend, [req])
    req = replace(
        req,
        compute_start=300,
        restore_end=300,
        compute_end=530,
        allocation_generation=generation,
    )
    if mutation == "configs":
        req.request_configs["options"]["ids"].append(2)
        assert backend._history[req.request_id].request_configs == {
            "options": {"ids": [1]}
        }
    elif mutation == "tokens":
        req = replace(req, token_ids=(999, *req.token_ids[1:]))
    else:
        process = engine.token_database.process_tokens

        def inexact(*args: Any, **kwargs: Any) -> list:
            plan = list(process(*args, **kwargs))
            start, end, key = plan[0]
            plan[0] = (
                (start + 1, end, key) if mutation == "start" else (start, end - 1, key)
            )
            return plan

        monkeypatch.setattr(engine.token_database, "process_tokens", inexact)
    calls = list(engine.gpu_connector.calls)
    with pytest.raises(ValueError, match="retained prefix|exact chunks"):
        backend.bind_step([req], _registry(engine))
    assert engine.gpu_connector.calls == calls
