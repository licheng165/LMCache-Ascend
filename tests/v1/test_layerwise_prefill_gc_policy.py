# SPDX-License-Identifier: Apache-2.0
"""Plan C worker GC policy modes (0911-2)."""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillFenceError,
    _apply_layerwise_gc_policy,
    _layerwise_gc_mode,
)

# Local
from tests.v1.test_layerwise_prefill_async import _backend, _execute, _request
from tests.v1.test_layerwise_prefill_sync import runtime  # noqa: F401
from tests.v1.test_layerwise_prefill_sync_pages import page_runtime  # noqa: F401

import gc


@pytest.fixture
def policy_runtime(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("runtime")


@pytest.fixture
def policy_page_runtime(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("page_runtime")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("default", "default"),
        ("DEFAULT", "default"),
        (" freeze ", "freeze"),
        ("Freeze", "freeze"),
        ("STEPWISE", "stepwise"),
        (" stepwise ", "stepwise"),
    ],
)
def test_mode_parsing_accepts_case_and_whitespace(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", raw)
    assert _layerwise_gc_mode() == expected


def test_mode_defaults_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_GC_MODE", raising=False)
    assert _layerwise_gc_mode() == "default"


def test_invalid_mode_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", "fast")
    with pytest.raises(ValueError, match="freeze"):
        _apply_layerwise_gc_policy()


def test_default_mode_leaves_gc_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_GC_MODE", raising=False)
    threshold, enabled = gc.get_threshold(), gc.isenabled()
    assert _apply_layerwise_gc_policy() == "default"
    assert gc.get_threshold() == threshold and gc.isenabled() == enabled


def test_freeze_mode_sets_threshold_and_reports_mode(
    policy_page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", "freeze")
    threshold = gc.get_threshold()
    try:
        engine = policy_page_runtime.engine()
        backend = _backend(engine)
        assert backend._gc_mode == "freeze"
        assert gc.isenabled()  # Young collections keep their cadence.
        assert gc.get_threshold() == (700, 10, 1000)
        _execute(engine, backend, [_request()])
        stats = backend.window_stats()
        assert stats["gc"]["mode"] == "freeze"
        assert stats["gc"]["counts"][2] == 0  # gen2 effectively never fires.
    finally:
        gc.unfreeze()
        gc.set_threshold(*threshold)


def test_stepwise_mode_disables_and_collects_young_each_step(
    policy_page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lmcache_ascend.v1.layerwise_prefill_sync as sync_module
    from types import SimpleNamespace

    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", "stepwise")
    engine = policy_page_runtime.engine()
    backend = _backend(engine)
    assert backend._gc_mode == "stepwise" and not gc.isenabled()
    collections: list[tuple] = []
    real_collect = gc.collect

    def observing(*args, **kwargs):
        collections.append(args)
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(
        sync_module,
        "gc",
        SimpleNamespace(collect=observing, disable=gc.disable, enable=gc.enable),
    )
    try:
        _execute(engine, backend, [_request()])
        stats = backend.window_stats()
        assert stats["gc"]["mode"] == "stepwise"
        # Exactly one bounded young collection per step, after the commit gate.
        assert collections == [(0,)]
    finally:
        gc.enable()


def test_policy_applies_to_sync_p_backend_too(
    policy_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", "stepwise")
    threshold = gc.get_threshold()
    try:
        engine = policy_runtime.engine()
        backend = engine.layerwise_prefill_window_backend
        assert backend._gc_mode == "stepwise" and not gc.isenabled()
    finally:
        gc.enable()
        gc.set_threshold(*threshold)


def test_fence_error_contract_unchanged_under_policy(
    policy_page_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_GC_MODE", "freeze")
    threshold = gc.get_threshold()
    try:
        engine = policy_page_runtime.engine()
        backend = _backend(engine)
        backend._unsafe_transfer = True
        backend._failure_error = LayerwisePrefillFenceError("sentinel")
        with pytest.raises(LayerwisePrefillFenceError, match="Cannot release"):
            backend.abort_step()
    finally:
        gc.unfreeze()
        gc.set_threshold(*threshold)
