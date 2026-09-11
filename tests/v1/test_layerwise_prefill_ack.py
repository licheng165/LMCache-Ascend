# SPDX-License-Identifier: Apache-2.0
"""Full-identity ACK transport and host statistics, with CPU TP and real Gloo."""

# ruff: noqa: F811

# Standard
from collections import Counter, OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
import multiprocessing
import pickle
import struct

# Third Party
from lmcache.utils import CacheEngineKey
import pytest
import torch
import vllm.distributed.parallel_state

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.layerwise_prefill_sync import LayerwisePrefillFenceError

# Local
from tests.v1.test_layerwise_prefill_async import _backend, _execute, _tp_backends
from tests.v1.test_layerwise_prefill_sync import _request, runtime  # noqa: F401
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401


class Unpickleable:
    def __reduce__(self) -> Any:
        raise RuntimeError("cannot pickle this identity")


class UnprintableError(ValueError):
    def __str__(self) -> str:
        raise RuntimeError("cannot print this error")


@dataclass
class PickleOnce:
    value: int
    calls: int = field(default=0, compare=False)

    def __reduce__(self) -> Any:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("serialized twice before collective")
        return type(self), (self.value,)


def _key(configs: dict) -> CacheEngineKey:
    return CacheEngineKey("ack-model", 2, 0, 123, torch.bfloat16, configs)


@pytest.fixture
def ack_runtime(runtime: Any, monkeypatch: Any) -> Any:
    engines = [runtime.engine(size=2)]
    engines.append(runtime.engine(1, engines[0], size=2))
    counts = [Counter(), Counter()]
    pointers = [[], []]
    for name in ("all_gather", "all_gather_object"):
        original = getattr(torch.distributed, name)

        def gather(
            out: list,
            status: Any,
            group: Any,
            *,
            name: str = name,
            original: Any = original,
        ) -> None:
            rank = runtime.thread.rank
            counts[rank][name] += 1
            if name == "all_gather":
                assert status.shape == (4096,) and status.dtype == torch.uint8
                assert all(
                    t.device.type == "cpu" and not t.is_inference()
                    for t in [status, *out]
                )
                pointers[rank].append(tuple(t.data_ptr() for t in [status, *out]))
            original(out, status, group=group)

        monkeypatch.setattr(torch.distributed, name, gather)
    return SimpleNamespace(
        engines=engines,
        counts=counts,
        pointers=pointers,
        runtime=runtime,
    )


def _ack_pair(
    env: Any,
    identities: list,
    errors: tuple = (None, None),
    failure: type[Exception] | None = None,
    match: str = "",
) -> None:
    def run(rank: int) -> None:
        with torch.inference_mode():
            if failure is None:
                env.engines[rank].layerwise_prefill_ack(identities[rank], errors[rank])
            else:
                with pytest.raises(failure, match=match):
                    env.engines[rank].layerwise_prefill_ack(
                        identities[rank], errors[rank]
                    )

    env.runtime.parallel(lambda: run(0), lambda: run(1))


def test_full_python_equality_not_pickle_bytes(ack_runtime: Any) -> None:
    configs = [{"a": 1, "b": [2, 3]}, {"b": [2, 3], "a": 1}]
    identities = [
        (1, ("save", replace(_request(), request_configs=config), _key(config)))
        for config in configs
    ]
    assert identities[0] == identities[1]
    assert pickle.dumps(identities[0]) != pickle.dumps(identities[1])
    _ack_pair(ack_runtime, identities)
    _ack_pair(ack_runtime, identities)
    for rank, engine in enumerate(ack_runtime.engines):
        stats = engine.layerwise_prefill_ack_stats()
        size = len(
            pickle.dumps(
                [(identities[rank], None, False)], protocol=pickle.HIGHEST_PROTOCOL
            )
        )
        assert stats["count"] == 2 and stats["fast_count"] == 2
        assert stats["slow_count"] == 0
        assert stats["serialized_bytes"] == 2 * size
        assert stats["max_payload_bytes"] == size < 4096
        assert 0 < stats["max_ms"] <= stats["total_ms"]
        assert stats["by_phase"]["save"] == {
            k: v for k, v in stats.items() if k != "by_phase"
        }
        assert ack_runtime.counts[rank] == Counter(all_gather=2)
        assert ack_runtime.pointers[rank][0] == ack_runtime.pointers[rank][1]


def test_order_sensitive_python_identity_still_rejected(ack_runtime: Any) -> None:
    identities = [
        (1, ("load", OrderedDict(items)))
        for items in ([("a", 1), ("b", 2)], [("b", 2), ("a", 1)])
    ]
    _ack_pair(ack_runtime, identities, failure=ValueError, match="identity mismatch")
    assert all(c == Counter(all_gather=1) for c in ack_runtime.counts)


def test_one_peer_large_uses_same_exact_fallback(ack_runtime: Any) -> None:
    # Non-tag configs are deliberately excluded by CacheEngineKey.__eq__. They
    # still travel in full, and may make just one peer's equal key too large.
    identities = [("bind", _key({"arbitrary": value})) for value in ("", "x" * 8192)]
    assert identities[0] == identities[1]
    _ack_pair(ack_runtime, identities)
    for rank, engine in enumerate(ack_runtime.engines):
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == stats["slow_count"] == 1
        assert stats["fast_count"] == 0
        assert (stats["max_payload_bytes"] > 4096) == bool(rank)
        assert ack_runtime.counts[rank] == Counter(all_gather=1, all_gather_object=1)


def test_fallback_does_not_repeat_arbitrary_serializers(ack_runtime: Any) -> None:
    values = [PickleOnce(7), PickleOnce(7)]
    identities = [
        ("bind", values[rank], _key({"arbitrary": "x" * 8192 if rank else ""}))
        for rank in (0, 1)
    ]
    _ack_pair(ack_runtime, identities)
    assert [value.calls for value in values] == [1, 1]
    assert all(
        c == Counter(all_gather=1, all_gather_object=1) for c in ack_runtime.counts
    )


@pytest.mark.parametrize("overflow", [0, 1])
def test_capacity_includes_header(ack_runtime: Any, overflow: int) -> None:
    # Measure the fixed pickle overhead above the short-string encoding boundary.
    identity = (1, ("load", "x" * 300))
    overhead = (
        len(pickle.dumps([(identity, None, False)], protocol=pickle.HIGHEST_PROTOCOL))
        - 300
    )
    identity = (1, ("load", "x" * (4096 - 5 - overhead + overflow)))
    _ack_pair(ack_runtime, [identity, identity])
    for engine in ack_runtime.engines:
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["max_payload_bytes"] == 4091 + overflow
        # One byte over the packet no longer selects the object fallback: the
        # oversized payload agrees by digest in the same fixed-size gather.
        assert stats["slow_count"] == 0
        assert stats["fast_count"] == 1 and stats["digest_count"] == overflow


def test_large_equal_payload_agrees_by_digest_without_object_fallback(
    ack_runtime: Any,
) -> None:
    identity = ("bind", _key({"payload": "x" * 8192}))
    _ack_pair(ack_runtime, [identity, identity])
    for rank, engine in enumerate(ack_runtime.engines):
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == stats["fast_count"] == stats["digest_count"] == 1
        assert stats["slow_count"] == 0
        assert stats["max_payload_bytes"] > 4091
        assert ack_runtime.counts[rank] == Counter(all_gather=1)


def test_large_equal_objects_with_different_pickle_bytes_fall_back_and_pass(
    ack_runtime: Any,
) -> None:
    # Equal mappings with different insertion orders pickle differently: their
    # digests disagree and the exact object comparison still passes, so the
    # digest is only an optimization, never the agreement itself.
    configs = [
        {"a": 1, "pad": "x" * 8192, "b": [2, 3]},
        {"b": [2, 3], "pad": "x" * 8192, "a": 1},
    ]
    identities = [
        (1, ("save", replace(_request(), request_configs=config), _key(config)))
        for config in configs
    ]
    assert identities[0] == identities[1]
    assert pickle.dumps(identities[0]) != pickle.dumps(identities[1])
    _ack_pair(ack_runtime, identities)
    for rank, engine in enumerate(ack_runtime.engines):
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 1 and stats["fast_count"] == 0
        assert stats["digest_count"] == 0 and stats["slow_count"] == 1
        assert stats["max_payload_bytes"] > 4091
    assert all(
        c == Counter(all_gather=1, all_gather_object=1) for c in ack_runtime.counts
    )


def test_tampered_digest_falls_back_to_exact_comparison(
    ack_runtime: Any, monkeypatch: Any
) -> None:
    original = torch.distributed.all_gather

    def gather(out: list, packet: Any, group: Any) -> None:
        if ack_runtime.runtime.thread.rank == 1:
            with torch.inference_mode(False):
                packet = packet.clone()
            packet[5] ^= 0xFF  # A well-formed digest packet with wrong bytes.
            original(out, packet, group=group)
            return
        original(out, packet, group=group)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    _ack_pair(ack_runtime, [("bind", _key({"payload": "x" * 8192}))] * 2)
    assert all(
        c == Counter(all_gather=1, all_gather_object=1) for c in ack_runtime.counts
    )
    for engine in ack_runtime.engines:
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 1 and stats["digest_count"] == 0
        assert stats["slow_count"] == 1


def test_batched_acknowledgements_share_one_collective(ack_runtime: Any) -> None:
    appends = [(1, ("validate", "load")), (1, ("prepare_load", "row"))]

    def run(rank: int) -> None:
        engine = ack_runtime.engines[rank]
        for identity in appends:
            engine.layerwise_prefill_ack(identity, flush=False)
        engine.layerwise_prefill_ack((1, ("load_ready", "row")))

    ack_runtime.runtime.parallel(lambda: run(0), lambda: run(1))
    for engine in ack_runtime.engines:
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 3 and stats["fast_count"] == 1
        assert stats["slow_count"] == 0
        for phase in ("validate", "prepare_load"):
            assert stats["by_phase"][phase]["count"] == 1
            assert stats["by_phase"][phase]["fast_count"] == 0
            assert stats["by_phase"][phase]["serialized_bytes"] == 0
        ready = stats["by_phase"]["load_ready"]
        assert ready["count"] == 1 and ready["fast_count"] == 1
        assert 0 < ready["max_payload_bytes"] < 4091
    assert all(c == Counter(all_gather=1) for c in ack_runtime.counts)


def test_batched_sequence_divergence_fails_on_all_ranks(ack_runtime: Any) -> None:
    def run(rank: int) -> None:
        engine = ack_runtime.engines[rank]
        engine.layerwise_prefill_ack((1, ("validate", "load")), flush=False)
        if rank:
            engine.layerwise_prefill_ack((1, ("prepare_load", "extra")), flush=False)
        with pytest.raises(ValueError, match="identity mismatch"):
            engine.layerwise_prefill_ack((1, ("load_ready",)))

    ack_runtime.runtime.parallel(lambda: run(0), lambda: run(1))
    assert all(c == Counter(all_gather=1) for c in ack_runtime.counts)


def test_error_status_rides_the_next_flush(ack_runtime: Any) -> None:
    def run(rank: int) -> None:
        engine = ack_runtime.engines[rank]
        engine.layerwise_prefill_ack((1, ("validate", "load")), flush=False)
        engine.layerwise_prefill_ack(
            (1, ("save", "row")),
            ValueError("rank-one failure") if rank else None,
            # Errors ride the batch: row errors already reach peers through the
            # shared-handle envelope broadcast, and an immediate flush would
            # strand healthy ranks at that broadcast rendezvous.
            flush=False,
        )
        with pytest.raises(ValueError, match="rank-one failure|identity mismatch"):
            engine.layerwise_prefill_ack((1, ("load_ready",)))

    ack_runtime.runtime.parallel(lambda: run(0), lambda: run(1))
    assert all(c == Counter(all_gather=1) for c in ack_runtime.counts)


def test_batch_overflow_agrees_by_digest(ack_runtime: Any) -> None:
    # Distinct payloads: pickle memoization would collapse identical strings.
    bigs = [(1, ("validate", "load", "x" * 2500)), (1, ("prepare_load", "y" * 2500))]

    def run(rank: int) -> None:
        engine = ack_runtime.engines[rank]
        for identity in bigs:
            engine.layerwise_prefill_ack(identity, flush=False)
        engine.layerwise_prefill_ack((1, ("load_ready",)))

    ack_runtime.runtime.parallel(lambda: run(0), lambda: run(1))
    for engine in ack_runtime.engines:
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 3 and stats["slow_count"] == 0
        assert stats["fast_count"] == stats["digest_count"] == 1
        assert stats["max_payload_bytes"] > 4091
    assert all(c == Counter(all_gather=1) for c in ack_runtime.counts)


def test_buffer_cache_tracks_group_and_world_without_tensor_item(
    monkeypatch: Any,
) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = SimpleNamespace(world_size=2)
    groups = [object(), object()]
    tp = SimpleNamespace(cpu_group=groups[0])
    monkeypatch.setattr(vllm.distributed.parallel_state, "get_tp_group", lambda: tp)
    buffers = []

    def gather(out: list, packet: Any, group: Any) -> None:
        assert group is tp.cpu_group and len(out) == engine.metadata.world_size
        assert all(
            t.device.type == "cpu" and not t.is_inference() for t in [packet, *out]
        )
        buffers.append((packet, *out))
        for peer in out:
            peer.copy_(packet)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    with monkeypatch.context() as host_only:
        for name in ("item", "__getitem__"):
            host_only.setattr(torch.Tensor, name, lambda *a: pytest.fail("tensor item"))
        for group, size in (
            (groups[0], 2),
            (groups[0], 2),
            (groups[1], 2),
            (groups[1], 3),
        ):
            tp.cpu_group, engine.metadata.world_size = group, size
            with torch.inference_mode():
                engine.layerwise_prefill_ack((1, ("source_done", "private key")))
    assert buffers[0][0] is buffers[1][0]
    assert buffers[1][0] is not buffers[2][0]
    assert buffers[2][0] is not buffers[3][0]
    assert all(t.numpy().tobytes() == bytes(4096) for batch in buffers for t in batch)


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("opposing", [False, True])
@pytest.mark.parametrize("fence", [False, True])
def test_nonzero_peer_error_and_opposing_phase(
    ack_runtime: Any,
    large: bool,
    opposing: bool,
    fence: bool,
) -> None:
    identities = [(1, ("source_done", "same")), (1, ("source_done", "same"))]
    if opposing:
        identities[1] = (1, ("abort_devices",))
    error_type = LayerwisePrefillFenceError if fence else ValueError
    error = error_type("rank-one failure" + ("x" * 8192 if large else ""))
    _ack_pair(ack_runtime, identities, (None, error), error_type, "rank-one failure")
    for rank, engine in enumerate(ack_runtime.engines):
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 1 and stats["total_ms"] > 0
        assert stats["slow_count"] == int(large)
        assert ack_runtime.counts[rank] == Counter(
            all_gather=1, all_gather_object=int(large)
        )


@pytest.mark.parametrize("kind", ["identity", "error_string", "fence_identity"])
def test_serialization_failure_is_error_status_on_all_ranks(
    ack_runtime: Any, kind: str
) -> None:
    identities = [(1, ("source_done",)), (1, ("source_done",))]
    errors = (None, UnprintableError()) if kind == "error_string" else (None, None)
    if kind != "error_string":
        identities[1] = (1, ("source_done", Unpickleable()))
    if kind == "fence_identity":
        errors = (None, LayerwisePrefillFenceError("unknown device"))
    failure = LayerwisePrefillFenceError if kind == "fence_identity" else ValueError
    _ack_pair(
        ack_runtime, identities, errors, failure, "ACK status serialization failed"
    )
    assert all(
        c == Counter(all_gather=1, all_gather_object=1) for c in ack_runtime.counts
    )
    assert all(
        e.layerwise_prefill_ack_stats()["slow_count"] == 1 for e in ack_runtime.engines
    )


@pytest.mark.parametrize(
    "flag,length",
    [(0, 20), (2, 20), (1, 0), (1, 5), (1, 4097), (1, 2**32 - 1), (1, 4096)],
)
def test_malformed_or_incomplete_packet_collectively_falls_back(
    ack_runtime: Any,
    monkeypatch: Any,
    flag: int,
    length: int,
) -> None:
    original = torch.distributed.all_gather

    def gather(out: list, packet: Any, group: Any) -> None:
        if ack_runtime.runtime.thread.rank == 1:
            with torch.inference_mode(False):
                packet = packet.clone()
            struct.pack_into("!BI", packet.numpy(), 0, flag, length)
        original(out, packet, group=group)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    _ack_pair(ack_runtime, [(1, ("load_ready",))] * 2)
    assert all(
        c == Counter(all_gather=1, all_gather_object=1) for c in ack_runtime.counts
    )


@pytest.mark.parametrize("slow", [False, True])
def test_transport_failure_is_fence_without_another_collective(
    ack_runtime: Any,
    monkeypatch: Any,
    slow: bool,
) -> None:
    name = "all_gather_object" if slow else "all_gather"
    calls = [0, 0]

    def broken(*args: Any, **kwargs: Any) -> None:
        calls[ack_runtime.runtime.thread.rank] += 1
        raise RuntimeError("transport broken")

    monkeypatch.setattr(torch.distributed, name, broken)
    backends = []
    for rank, engine in enumerate(ack_runtime.engines):
        ack_runtime.runtime.thread.rank = rank
        backends.append(_backend(engine))
    for backend in backends:
        backend._step_future = Future()

    def run(rank: int) -> None:
        backend = backends[rank]
        with pytest.raises(LayerwisePrefillFenceError, match="acknowledgement failed"):
            backend._ack(
                (
                    "source_done",
                    ("x" if rank else "y") * 8192 if slow else "",
                )
            )
        assert isinstance(backend._step_future.exception(), LayerwisePrefillFenceError)
        assert backend._unsafe_transfer and not backend._abort_drained

    ack_runtime.runtime.parallel(lambda: run(0), lambda: run(1))
    assert calls == [1, 1]
    for engine in ack_runtime.engines:
        stats = engine.layerwise_prefill_ack_stats()
        assert stats["count"] == 1 and stats["total_ms"] > 0
        assert set(stats["by_phase"]) == {"source_done"}
        assert stats["slow_count"] == int(slow)


def test_zero_reset_snapshot_and_bounded_phase_names(monkeypatch: Any) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    zero = engine.layerwise_prefill_ack_stats()
    assert zero == dict(
        count=0,
        total_ms=0.0,
        max_ms=0.0,
        serialized_bytes=0,
        max_payload_bytes=0,
        fast_count=0,
        digest_count=0,
        slow_count=0,
        by_phase={},
    )
    engine.metadata = SimpleNamespace(world_size=1)
    # Host-only and safe without any process group (including fake window engines).
    for name in ("all_gather", "all_gather_object", "is_initialized"):
        monkeypatch.setattr(
            torch.distributed, name, lambda *a, **k: pytest.fail("device/collective")
        )
    for identity in [
        ("bind", 1),
        (2, ("window_bind", 3)),
        ("failed_step_drained", 2),
        ("opaque-step", ("source_done",)),
        None,
        (),
        ({},),
    ]:
        engine.layerwise_prefill_ack(identity)
    for index in range(30):
        engine.layerwise_prefill_ack((1, (f"arbitrary-{index}-" + "x" * 5000,)))
    with pytest.raises(ValueError, match="ordinary"):
        engine.layerwise_prefill_ack((2, ("save",)), ValueError("ordinary"))
    snapshot = engine.layerwise_prefill_ack_stats()
    assert snapshot["count"] == 38
    assert (
        snapshot["serialized_bytes"]
        == snapshot["fast_count"]
        == snapshot["slow_count"]
        == 0
    )
    assert set(snapshot["by_phase"]) == {
        "bind",
        "window_bind",
        "failed_step_drained",
        "source_done",
        "save",
        "other",
    }
    snapshot["by_phase"]["save"]["count"] = -1
    snapshot["count"] = -1
    assert engine.layerwise_prefill_ack_stats()["by_phase"]["save"]["count"] == 1
    engine.reset_layerwise_prefill_ack_stats()
    assert engine.layerwise_prefill_ack_stats() == zero


def test_failed_calls_accumulate_host_time_in_finally(monkeypatch: Any) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = SimpleNamespace(world_size=1)
    ticks = iter((1.0, 1.003, 2.0, 2.007))
    monkeypatch.setattr(
        "lmcache_ascend.v1.cache_engine.time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )
    engine.layerwise_prefill_ack((1, ("save",)))
    with pytest.raises(ValueError):
        engine.layerwise_prefill_ack((1, ("save",)), ValueError("failed"))
    stats = engine.layerwise_prefill_ack_stats()
    assert stats["count"] == stats["by_phase"]["save"]["count"] == 2
    assert stats["total_ms"] == pytest.approx(10)
    assert stats["max_ms"] == pytest.approx(7)


def test_actual_async_two_steps_packet_counts(
    page_runtime: Any, monkeypatch: Any
) -> None:
    engines, backends = _tp_backends(page_runtime)
    counts = [Counter(), Counter()]
    for name in ("all_gather", "all_gather_object"):
        original = getattr(torch.distributed, name)

        def gather(
            out: list,
            status: Any,
            group: Any,
            *,
            name: str = name,
            original: Any = original,
        ) -> None:
            counts[page_runtime.thread.rank][name] += 1
            original(out, status, group=group)

        monkeypatch.setattr(torch.distributed, name, gather)
    for start, end in ((0, 300), (300, 530)):
        for rank, engine in enumerate(engines):
            engine.reset_layerwise_prefill_ack_stats()
            counts[rank].clear()
        page_runtime.parallel(
            *[
                lambda rank=rank, start=start, end=end: _execute(
                    engines[rank], backends[rank], [_request(start=start, end=end)]
                )
                for rank in (0, 1)
            ]
        )
        for rank, engine in enumerate(engines):
            stats = engine.layerwise_prefill_ack_stats()
            phases = stats["by_phase"]
            # Plan A: rows issue no acknowledgements; only the five per-step
            # gates (window_bind, bind, device_finish, finish, commit) enter a
            # collective.
            assert stats["count"] == 5
            assert set(phases) == {
                "window_bind",
                "bind",
                "device_finish",
                "finish",
                "commit",
            }
            for phase in phases.values():
                assert phase["count"] == 1
            assert counts[rank]["all_gather"] == stats["fast_count"] == 5
            assert counts[rank]["all_gather_object"] == stats["slow_count"] == 0
            assert stats["count"] == stats["fast_count"] + stats["slow_count"]


@pytest.mark.parametrize("failure", ["error", "serialization", "opposing_phase"])
def test_real_two_process_gloo_full_status_and_failed_handshake(
    runtime: Any,
    tmp_path: Any,
    failure: str,
) -> None:
    engines = [runtime.engine(size=2)]
    engines.append(runtime.engine(1, engines[0], size=2))
    context = multiprocessing.get_context("fork")
    results = context.Queue()

    def worker(rank: int, engine: Any) -> None:
        try:
            torch.distributed.is_initialized = runtime.real_initialized
            torch.distributed.all_gather_object = runtime.real_gather
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{tmp_path}/ack-gloo",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=20),
            )
            vllm.distributed.parallel_state.get_tp_group = lambda: SimpleNamespace(
                cpu_group=torch.distributed.group.WORLD,
                world_size=2,
                rank_in_group=rank,
            )
            counts = Counter()
            original = torch.distributed.all_gather

            def gather(*args: Any, **kwargs: Any) -> None:
                counts["tensor"] += 1
                # This is still the fixture mock: real groups must dispatch to Gloo.
                original(*args, **kwargs)

            torch.distributed.all_gather = gather
            torch.distributed.distributed_c10d.all_gather = gather
            configs = dict([("a", 1), ("b", 2)][:: 1 if rank == 0 else -1])
            for large in (False, True):
                config = {**configs, "arbitrary": "x" * 8192 if large and rank else ""}
                identity = (
                    "bind",
                    replace(_request(), request_configs=configs),
                    _key(config),
                )
                before = counts["tensor"]
                with torch.inference_mode():
                    engine.layerwise_prefill_ack(identity)
                assert counts["tensor"] - before == (3 if large else 1)
            backend = _backend(engine)
            backend._step_future = Future()
            identity = ("source_done",)
            error = None
            if rank == 1:
                if failure == "serialization":
                    identity = ("source_done", Unpickleable())
                elif failure == "opposing_phase":
                    identity = ("abort_devices", "x" * 8192)
                else:
                    error = ValueError("rank-one future failure")
            with pytest.raises(ValueError):
                backend._ack(identity, error)
            assert isinstance(backend._step_future.exception(), ValueError)
            assert backend._abort_drained and not backend._unsafe_transfer
            stats = engine.layerwise_prefill_ack_stats()
            assert stats["count"] == 4
            assert stats["slow_count"] == (1 if failure == "error" else 2)
            assert stats["by_phase"]["bind"]["count"] == 2
            assert stats["by_phase"]["failed_step_drained"]["count"] == 1
            assert stats["count"] + 2 * stats["slow_count"] == counts["tensor"]
            results.put((rank, stats["count"], counts["tensor"]))
        finally:
            if runtime.real_initialized():
                torch.distributed.destroy_process_group()

    processes = [
        context.Process(target=worker, args=(rank, engine))
        for rank, engine in enumerate(engines)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=40)
            assert process.exitcode == 0
        outcomes = sorted(results.get(timeout=5) for _ in processes)
        operations = 6 if failure == "error" else 8
        assert outcomes == [(0, 4, operations), (1, 4, operations)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
