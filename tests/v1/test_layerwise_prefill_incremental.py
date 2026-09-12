# SPDX-License-Identifier: Apache-2.0
"""Incremental bind caches reproduce full recomputation exactly (0911-2)."""

# Standard
from typing import Any

# Third Party
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
import pytest
import torch

# Local
from tests.v1.test_layerwise_prefill_sync import (  # noqa: F401
    _slots,
    _step,
    runtime,
)


@pytest.fixture
def inc_runtime(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("runtime")


def _long_request(
    start: int = 0,
    end: int = 300,
    generation: int = 1,
    request_id: str = "req-0",
) -> LayerwisePrefillRequest:
    return LayerwisePrefillRequest(
        request_id=request_id,
        allocation_generation=generation,
        token_ids=tuple(range(end + 700)),
        compute_start=start,
        compute_end=end,
        restore_end=start,
        block_ids_by_bank=(
            (tuple(range(1, 14)), tuple(range(14, 27))),
            (tuple(range(27, 40)), tuple(range(40, 53))),
        ),
        block_size=128,
        request_configs={"lmcache.tag.tenant": "sentinel"},
    )


def _bind(engine: Any, backend: Any, request: LayerwisePrefillRequest) -> None:
    # A full step between binds: the incremental caches must survive real
    # bind/row/finish cycles, not bare bind calls.
    _step(engine, backend, [request])


def _reference_plan(
    engine: Any, request: LayerwisePrefillRequest, group: int, end: int
):
    return [
        (start, stop, key.with_new_worker_id(0))
        for start, stop, key in engine.token_database.process_tokens(
            list(request.token_ids[:end]),
            kv_group=group,
            request_configs=request.request_configs,
        )
    ]


def test_incremental_plans_and_slots_match_full_recompute(
    inc_runtime: Any,
) -> None:
    engine = inc_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    # Production continuations keep one allocation generation for the whole
    # request (0911-1 log); the caches key on it and extend step over step.
    spans = ((0, 300), (300, 530), (530, 1024), (1024, 1300), (1300, 1500))
    for start, end in spans:
        request = _long_request(start, end, generation=1)
        _bind(engine, backend, request)
        for group in (0, 1):
            watermark, cached, _ = backend._plan_cache[request.request_id, 1, group]
            # The cached aligned prefix must equal a full recomputation; the
            # step's own manifest validation proved the tail chunks in use.
            assert watermark == end - end % 256
            assert list(cached) == _reference_plan(engine, request, group, watermark)
            # Re-querying any smaller extent, aligned or not, stays exact.
            for extent in {start, end}:
                if 0 < extent <= watermark:
                    assert backend._bind_plan(request, group, extent) == (
                        _reference_plan(engine, request, group, extent)
                    )
        for bank in (0, 1):
            for group in (0, 1):
                extent, tensor = backend._slot_cache[request.request_id, 1, bank, group]
                assert extent == end
                assert torch.equal(tensor, _slots(request, bank, group, end))
        assert all(entry[0] % 256 == 0 for entry in backend._plan_cache.values())


def test_generation_change_recomputes_and_requests_are_isolated(
    inc_runtime: Any,
) -> None:
    engine = inc_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    first = _long_request(0, 300, generation=1)
    _bind(engine, backend, first)
    second = _long_request(300, 530, generation=2)
    _bind(engine, backend, second)
    assert set(backend._plan_cache) == {
        (second.request_id, second.allocation_generation, group) for group in (0, 1)
    }
    assert set(backend._slot_cache) == {
        (second.request_id, second.allocation_generation, bank, group)
        for bank in (0, 1)
        for group in (0, 1)
    }
    for group in (0, 1):
        watermark, cached, _ = backend._plan_cache[second.request_id, 2, group]
        assert list(cached) == _reference_plan(engine, second, group, watermark)


def test_abort_request_drops_incremental_caches(inc_runtime: Any) -> None:
    engine = inc_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    request = _long_request(0, 300)
    _bind(engine, backend, request)
    assert backend._plan_cache and backend._slot_cache
    backend.abort_request(request.request_id)
    assert not backend._plan_cache and not backend._slot_cache


def test_continuation_processes_only_new_tokens(inc_runtime: Any) -> None:
    engine = inc_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    lengths: list[int] = []
    process = engine.token_database.process_tokens

    def recording(tokens, **kwargs):
        lengths.append(len(tokens))
        return process(tokens, **kwargs)

    engine.token_database.process_tokens = recording
    _bind(engine, backend, _long_request(0, 300, generation=1))
    first = list(lengths)
    _bind(engine, backend, _long_request(300, 530, generation=1))
    # One call per (group, extent): the first bind hashes [0, 300); the
    # continuation re-derives the 44-token tail slice and hashes only the
    # new [256, 530) range per group.
    assert first == [300, 300]
    assert sorted(lengths[2:]) == [44, 44, 274, 274]


def test_unaligned_tail_never_pollutes_the_cache(inc_runtime: Any) -> None:
    engine = inc_runtime.engine()
    backend = engine.layerwise_prefill_window_backend
    first = _long_request(0, 300, generation=1)
    _bind(engine, backend, first)
    second = _long_request(300, 530, generation=1)
    _bind(engine, backend, second)
    for identity, (extent, _, _) in backend._plan_cache.items():
        assert identity[0] == "req-0" and identity[2] in (0, 1)
        # 530's partial chunk [512, 530) stays outside the high-water cache.
        assert extent == 512
    assert backend._bind_plan(second, 0, 530) == _reference_plan(engine, second, 0, 530)
