# SPDX-License-Identifier: Apache-2.0
"""Real sync backend/connector roundtrips with CPU-backed NPU/native mocks."""

# ruff: noqa: F811

# Standard
from dataclasses import replace
from typing import Any
import weakref

# Third Party
import pytest
import torch

# Local
from tests.v1.test_layerwise_prefill_row_transfer import row_env  # noqa: F401
from tests.v1.test_layerwise_prefill_sync import (
    _metadata,
    _request,
    _sentinel,
    _slots,
    _view,
    runtime,  # noqa: F401
)


@pytest.mark.parametrize("generation", [1, 2])
def test_real_connector_reuses_four_bind_plans_across_all_rows(
    runtime: Any, row_env: Any, monkeypatch: pytest.MonkeyPatch, generation: int
) -> None:
    engine = runtime.engine()
    connector = row_env.connector
    # The row fixture forbids initialization inside a row hook. Here the real
    # backend must initialize both full groups before calling the real preparer.
    monkeypatch.delattr(connector, "initialize_kvcaches_ptr")
    monkeypatch.delattr(connector, "_lazy_initialize_buffer")
    connector._group_layouts = {}
    connector._dsa_kv_topology_view = _view()
    connector.lmcache_chunk_size = 256
    planes = {group: row_env.planes(group, blocks=16) for group in (0, 1)}
    registry = {
        row[0]: planes[group]
        for group, rows in enumerate(_view().rows_by_group)
        for row in rows
    }
    engine.gpu_connector = connector
    backend = engine.layerwise_prefill_window_backend
    prepare = connector.prepare_layerwise_prefill_slots
    transfer = connector.transfer_layerwise_prefill_row
    prepared = []

    def prepare_slots(
        mapping: torch.Tensor,
        *,
        kv_group: int,
        capacity: int,
        identity: Any = None,
    ) -> Any:
        assert [connector.get_num_layers(group) for group in (0, 1)] == [79, 22]
        assert mapping.device.type == "cpu" and mapping.dtype == torch.long
        assert capacity == 64
        plan = prepare(mapping, kv_group=kv_group, capacity=capacity, identity=identity)
        assert not isinstance(plan, torch.Tensor)
        prepared.append(weakref.ref(plan))
        return plan

    def transfer_row(*args: Any, kv_group: int, direction: bool, timing: dict) -> None:
        assert len(args) == 5
        assert args[4] is backend._slots[req.request_id, key[4], kv_group]
        assert timing is backend._transfer_timings[int(direction)]
        host_ops = row_env.host_ops.copy()
        transfer(*args, kv_group=kv_group, direction=direction, timing=timing)
        assert not row_env.pending
        # The copy-aware mock counts pointer/offset/size factories as uploads
        # too. Only those three tables may be uploaded; slots remain prepared.
        assert row_env.host_ops["uploads"] == host_ops["uploads"] + 3
        for operation in (
            "readback",
            "cat",
            "tolist",
            "validation_scans",
            "unique_checks",
        ):
            assert row_env.host_ops[operation] == host_ops[operation], operation

    monkeypatch.setattr(connector, "prepare_layerwise_prefill_slots", prepare_slots)
    monkeypatch.setattr(connector, "transfer_layerwise_prefill_row", transfer_row)
    for step, (start, end) in enumerate(((0, 8), (8, 14)), start=1):
        req = replace(
            _request(start=start, end=end, generation=generation if start else 1),
            block_size=4,
        )
        if start and generation == 2:
            req = replace(
                req,
                block_ids_by_bank=tuple(
                    tuple(tuple(block + 2 for block in blocks) for blocks in groups)
                    for groups in req.block_ids_by_bank
                ),
            )
        backend.bind_step([req], registry)
        assert len(prepared) == 4 * step
        assert all(ref() is None for ref in prepared[:-4])
        assert all(ref() is not None for ref in prepared[-4:])
        assert row_env.host_ops["uploads"] == 4 * step + 3 * len(row_env.calls)
        before = len(row_env.calls)
        for _, latent, indexer in _view().executions:
            for key in (latent, indexer):
                if key is None:
                    continue
                group, row, bank = key[2:]
                metadata = _metadata(_view(), key, [req])
                backend.wait_for_load(metadata)
                slots = _slots(req, bank, group, end)
                for index, plane in enumerate(registry[key[0]]):
                    flat = plane.as_subclass(torch.Tensor).view(64, -1)
                    expected = _sentinel(req, group, row, index, end)[:, None].expand(
                        end, flat.shape[1]
                    )
                    assert torch.equal(flat[slots[:start]], expected[:start])
                    flat[slots[start:]] = expected[start:]
                backend.sync_save(metadata, registry[key[0]])
                for plane in registry[key[0]]:
                    plane.fill_(-100)
        backend.finish_step()
        calls = row_env.calls[before:]
        assert sum(call["direction"] for call in calls) == 101
        assert sum(not call["direction"] for call in calls) == (101 if start else 0)
        assert all(
            call["sizes"] == [end if call["direction"] else start] for call in calls
        )
        assert len(prepared) == 4 * step and not backend._slots
        assert all(ref() is None for ref in prepared)
        assert row_env.host_ops["uploads"] == 4 * step + 3 * len(row_env.calls)
        assert sum(event[1] == "slots" for event in row_env.events) == 4 * step
        assert row_env.host_ops["readback"] == row_env.host_ops["cat"] == 0
    backend.abort_request(req.request_id)
