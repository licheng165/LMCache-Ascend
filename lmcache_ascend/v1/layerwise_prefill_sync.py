# SPDX-License-Identifier: Apache-2.0
"""Synchronous shared-bank P residency with request-owned, complete CPU rows.

This path deliberately has no transfer window, graph capture, or all-layer
device fallback. Row callbacks fence D2H and publish shared CPU handles on all
TP ranks. Page-first remote persistence completes only at the final step barrier.
"""

# Standard
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterator

# Third Party
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mooncake_layout import mooncake_page_layout_enabled
from lmcache.v1.pin_monitor import PinMonitor
import torch

# First Party
from lmcache_ascend.v1.dsa_kv_topology import DSAKVTopologyView

if TYPE_CHECKING:
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

logger = init_logger(__name__)


class LayerwisePrefillFenceError(RuntimeError):
    """An uncompleted device transfer requires worker restart, not slab reuse."""


@dataclass
class _RowPrefix:
    """Request-owned, trusted manifest adopted after all-TP completion ACK."""

    starts: list[int]
    ends: list[int]
    keys: list[CacheEngineKey]
    objects: list[MemoryObj]
    revision: int
    slab_generation: int


class LayerwisePrefillSyncBackend:
    """Production eager backend; construct with a validated Ascend engine/view.

    ``bind_step`` freezes scheduler bindings and the complete device registry.
    Each group's callbacks must visit its canonical rows in order. Manifests
    survive step and allocation-generation changes until ``abort_request``
    (also used for normal request cleanup). Errors poison the step; never retry
    a partially persisted step against recycled banks.
    """

    supports_sync_callbacks = True
    supports_transfer_window = False
    persists_indexer_group = True
    accepts_coordinator_validation_errors = True

    def __init__(self, engine: "AscendLMCacheEngine", view: DSAKVTopologyView) -> None:
        engine.validate_layerwise_prefill_sync(view)
        self._engine = engine
        self._view = view
        self._page_first = mooncake_page_layout_enabled(engine.config)
        self._step_future: Future | None = None
        self._commit_starts: dict[str, int] = {}
        self._pending_rows: dict[tuple[str, int, int, int], list[MemoryObj]] = {}
        self._required_backends = (
            ("RemoteBackend",) if engine.config.remote_url is not None else ()
        )
        self._requests: tuple[LayerwisePrefillRequest, ...] = ()
        self._history: dict[str, LayerwisePrefillRequest] = {}
        self._released: dict[str, int] = {}
        self._prefixes: dict[tuple[str, int, int, int], _RowPrefix] = {}
        self._caches: dict[str, tuple[torch.Tensor, ...]] = {}
        self._slots: dict[tuple[str, int, int], Any] = {}
        self._plans: dict[tuple[str, int, int], list[tuple]] = {}
        self._saved = [0, 0]
        self._ready: set[tuple[int, int]] = set()
        self._step = 0
        self._bound = False
        self._failed = False
        self._unsafe_transfer = False
        self._quarantined: list[MemoryObj] = []
        self._step_started = 0.0
        self._bind_seconds = 0.0
        self._load_seconds = 0.0
        self._save_seconds = 0.0
        self._reused_rows = 0
        self._published_handles = 0
        self._bind_parts = (0.0, 0.0, 0.0)
        self._ack_seconds: dict[str, float] = {}
        self._transfer_timings: tuple[dict[str, float], dict[str, float]] = ({}, {})
        self._transfer_timing_enabled = (
            getattr(
                engine.gpu_connector,
                "supports_layerwise_prefill_transfer_timings",
                False,
            )
            is True
        )
        self._prepared_slot_count = 0

    @property
    def topology_signature(self) -> str:
        """The frozen construction-path topology signature."""
        return self._view.signature

    def layer_count(self, group: int) -> int:
        """Return the authoritative group-local row count."""
        if group not in (0, 1):
            raise ValueError("Invalid layerwise-prefill group")
        return self._view.layer_counts[group]

    def bind_step(
        self,
        requests: list[LayerwisePrefillRequest],
        kv_caches: dict[str, Any],
        *,
        validation_error: Exception | None = None,
    ) -> None:
        """Freeze exact request/generation/range/bank bindings before any transfer.

        A continuation must match its retained prefix and cannot rewind tokens
        or generations. Both complete registry groups are initialized before
        shapes are queried or the public device hook is called.
        """
        error = None
        frozen = ()
        caches = {}
        slots = {}
        plans = {}
        commit_starts = {}
        plan_seconds = slot_seconds = 0.0
        prepared_count = 0
        started = perf_counter()
        try:
            if validation_error is not None:
                raise validation_error
            if self._bound or self._failed:
                raise ValueError(
                    "Previous layerwise-prefill step is unfinished or failed"
                )
            if mooncake_page_layout_enabled(self._engine.config) != self._page_first:
                raise ValueError(
                    "Layerwise-prefill page mode changed after capability freeze"
                )
            if any(not isinstance(req, LayerwisePrefillRequest) for req in requests):
                raise ValueError("Expected scheduler LayerwisePrefillRequest bindings")
            frozen = tuple(
                replace(req, request_configs=deepcopy(req.request_configs))
                for req in requests
            )
            if len({req.request_id for req in frozen}) != len(frozen):
                raise ValueError("Duplicate layerwise-prefill request binding")
            if self._engine.metadata.is_first_rank():
                logger.info(
                    "[PREFILL_SYNC_STEP] event=begin step=%d requests=%d "
                    "ranges=(request,generation,compute_start,compute_end,restore_end):%s",
                    self._step + 1,
                    len(frozen),
                    [
                        (
                            req.request_id,
                            req.allocation_generation,
                            req.compute_start,
                            req.compute_end,
                            req.restore_end,
                        )
                        for req in frozen
                    ],
                )
            for group, rows in enumerate(self._view.rows_by_group):
                for name, *_ in rows:
                    caches[name] = self._planes(kv_caches[name], group)
                    reference = caches[rows[0][0]]
                    if any(
                        plane.shape != expected.shape or plane.device != expected.device
                        for plane, expected in zip(caches[name], reference, strict=True)
                    ):
                        raise ValueError("Layerwise-prefill group KV geometry differs")
            occupied: dict[tuple[int, int], set[int]] = {}
            for req in frozen:
                if not isinstance(req.request_id, str) or any(
                    type(value) is not int
                    for value in (
                        req.allocation_generation,
                        req.compute_start,
                        req.compute_end,
                        req.restore_end,
                        req.block_size,
                    )
                ):
                    raise ValueError(
                        "Invalid layerwise-prefill request identity/ranges"
                    )
                if req.restore_end > req.compute_end:
                    raise ValueError(
                        "Restore extent exceeds this step's computed prefix"
                    )
                if req.allocation_generation <= self._released.get(req.request_id, 0):
                    raise ValueError("Released layerwise-prefill allocation generation")
                prior = self._history.get(req.request_id)
                commit_starts[req.request_id] = (
                    0 if prior is None else req.compute_start // 256 * 256
                )
                if prior is not None and (
                    req.allocation_generation < prior.allocation_generation
                    or req.compute_start != prior.compute_end
                    or req.restore_end != prior.compute_end
                    or req.token_ids[: prior.compute_end]
                    != prior.token_ids[: prior.compute_end]
                    or req.request_configs != prior.request_configs
                ):
                    raise ValueError(
                        "Layerwise-prefill continuation changed its retained prefix"
                    )
                if prior is not None and any(
                    (req.request_id, prior.allocation_generation, group, row)
                    not in self._prefixes
                    for group, count in enumerate(self._view.layer_counts)
                    for row in range(count)
                ):
                    raise ValueError("Missing retained layerwise-prefill manifest")
                if (
                    prior is not None
                    and req.allocation_generation == prior.allocation_generation
                ):
                    used_blocks = (
                        prior.compute_end + prior.block_size - 1
                    ) // prior.block_size
                    if req.block_size != prior.block_size or any(
                        req.block_ids_by_bank[bank][group][:used_blocks]
                        != prior.block_ids_by_bank[bank][group][:used_blocks]
                        for bank in (0, 1)
                        for group in (0, 1)
                    ):
                        raise ValueError(
                            "Changed bank allocation requires a new generation"
                        )
                stage_started = perf_counter()
                for group in (0, 1):
                    for end in {req.restore_end, req.compute_end}:
                        plan = (
                            list(
                                self._engine.token_database.process_tokens(
                                    list(req.token_ids[:end]),
                                    kv_group=group,
                                    request_configs=req.request_configs,
                                )
                            )
                            if end
                            else []
                        )
                        frontier = 0
                        for start, stop, _ in plan:
                            if start != frontier or stop != min(start + 256, end):
                                raise ValueError(
                                    "Token database did not return exact chunks"
                                )
                            frontier = stop
                        if frontier != end:
                            raise ValueError("Token database omitted the partial tail")
                        plans[req.request_id, group, end] = [
                            (start, stop, key.with_new_worker_id(0))
                            for start, stop, key in plan
                        ]
                plan_seconds += perf_counter() - stage_started
                stage_started = perf_counter()
                # Bank IDs originate on the host. Prepare their common position
                # arithmetic once, rather than reading device slots back per row.
                positions = torch.arange(
                    req.compute_end, dtype=torch.long, device="cpu"
                )
                block_indices = positions // req.block_size
                block_offsets = positions % req.block_size
                for bank, groups in enumerate(req.block_ids_by_bank):
                    for group, blocks in enumerate(groups):
                        used = occupied.setdefault((bank, group), set())
                        if len(set(blocks)) != len(blocks) or used.intersection(blocks):
                            raise ValueError("Layerwise-prefill request banks overlap")
                        used.update(blocks)
                        plane = caches[self._view.rows_by_group[group][0][0]][0]
                        if (
                            plane.shape[1] != req.block_size
                            or max(blocks) >= plane.shape[0]
                        ):
                            raise ValueError(
                                "Layerwise-prefill blocks exceed the KV plane"
                            )
                        block_tensor = torch.tensor(
                            blocks, dtype=torch.long, device="cpu"
                        )
                        slots[req.request_id, bank, group] = (
                            block_tensor[block_indices] * req.block_size + block_offsets
                        )
                slot_seconds += perf_counter() - stage_started
            self._engine.initialize_layerwise_prefill_layout(caches)
            prepare = getattr(
                self._engine.gpu_connector, "prepare_layerwise_prefill_slots", None
            )
            if callable(prepare):
                stage_started = perf_counter()
                for identity, mapping in slots.items():
                    group = identity[2]
                    plane = caches[self._view.rows_by_group[group][0][0]][0]
                    slots[identity] = prepare(
                        mapping,
                        kv_group=group,
                        capacity=int(plane.shape[0] * plane.shape[1]),
                    )
                    prepared_count += 1
                slot_seconds += perf_counter() - stage_started
        except Exception as exc:
            error = exc
        ack_started = perf_counter()
        self._engine.layerwise_prefill_ack(
            ("bind", self._step + 1, frozen, plans, self._page_first, commit_starts),
            error,
        )
        self._bind_parts = (plan_seconds, slot_seconds, perf_counter() - ack_started)
        for req in frozen:
            prior = self._history.get(req.request_id)
            if (
                prior is not None
                and prior.allocation_generation != req.allocation_generation
            ):
                for group, count in enumerate(self._view.layer_counts):
                    for row in range(count):
                        old = (req.request_id, prior.allocation_generation, group, row)
                        new = (req.request_id, req.allocation_generation, group, row)
                        self._prefixes[new] = self._prefixes.pop(old)
            self._history[req.request_id] = req
        self._requests, self._caches, self._slots = frozen, caches, slots
        self._plans = plans
        self._commit_starts = commit_starts
        self._step_future = Future() if self._page_first else None
        self._saved, self._ready = [0, 0], set()
        self._step += 1
        self._bound = True
        self._step_started = started
        self._bind_seconds = perf_counter() - started
        self._load_seconds = self._save_seconds = 0.0
        self._reused_rows = self._published_handles = 0
        self._prepared_slot_count = prepared_count
        self._ack_seconds = {}
        self._transfer_timings = ({}, {})

    def wait_for_load(
        self, metadata: Any, *, validation_error: Exception | None = None
    ) -> None:
        """Restore this row's full prefix on ALL TP ranks, including own chunks."""
        started = perf_counter()
        group, row, bank, name = self._validate(
            metadata, "load", validation_error=validation_error
        )
        if (group, row) in self._ready:
            self._load_seconds += perf_counter() - started
            return
        for req in self._requests:
            identity = (req.request_id, req.allocation_generation, group, row)
            prior = self._prefixes.get(identity)
            objects = prior.objects if prior is not None else None
            error = None
            try:
                if prior is None:
                    starts, ends, keys = self._plan(req, req.restore_end, group, row)
                    if self._page_first and self._engine.metadata.is_first_rank():
                        try:
                            if identity not in self._pending_rows:
                                sources = self._engine.resolve_layerwise_prefill_group(
                                    req.request_id,
                                    group,
                                    self._plans[req.request_id, group, req.restore_end],
                                    phase=f"prefill_load:{self._step}",
                                )
                                for i, source in enumerate(sources):
                                    self._pending_rows[
                                        req.request_id,
                                        req.allocation_generation,
                                        group,
                                        i,
                                    ] = source
                            objects = self._pending_rows.pop(identity)
                        except Exception as exc:
                            error = exc
                    objects = self._engine.resolve_layerwise_prefill_row(
                        *identity,
                        keys,
                        starts,
                        ends,
                        phase=f"prefill_load:{self._step}",
                        memory_objs=objects,
                        error=error,
                    )
                    self._published_handles += len(keys)
                else:
                    starts, ends, keys = prior.starts, prior.ends, prior.keys
                    self._reused_rows += 1
                if objects:
                    self._transfer(req, name, group, bank, objects, starts, ends, False)
            except Exception as exc:
                error = exc
            try:
                self._ack(("load", identity), error)
            except Exception:
                if objects is not None and prior is None:
                    if self._unsafe_transfer:
                        self._quarantined.extend(objects)
                    else:
                        self._release(objects)
                raise
            if prior is None:
                self._prefixes[identity] = _RowPrefix(
                    starts,
                    ends,
                    keys,
                    objects,
                    revision=self._step,
                    slab_generation=self._engine.shared_cpu_cache_generation,
                )
        self._ready.add((group, row))
        self._load_seconds += perf_counter() - started

    def sync_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
        *,
        validation_error: Exception | None = None,
    ) -> Future | None:
        """Fence and publish changed CPU rows; page mode returns the step commit future.

        The CPU source is complete synchronously, so banks may be reused on
        return. Layer-key mode also waits remote persistence here and returns None.
        """
        started = perf_counter()
        group, row, bank, name = self._validate(
            metadata, "save", kv_layer, validation_error=validation_error
        )
        root = self._engine.metadata.is_first_rank()
        for req in self._requests:
            identity = (req.request_id, req.allocation_generation, group, row)
            prior = self._prefixes[identity]
            starts, ends, keys = self._plan(req, req.compute_end, group, row)
            # Recompute the partial predecessor in its successor's exact format.
            # Full-hit bootstrap may restore one more token than compute_start.
            changed_start = (
                min(req.compute_start, prior.ends[-1] if prior.ends else 0) // 256 * 256
            )
            keep = changed_start // 256
            fresh = []
            objects = None
            error = None
            try:
                if keys[:keep] != prior.keys[:keep]:
                    raise ValueError("Layerwise-prefill unchanged chunk keys differ")
                if root:
                    for start, end in zip(starts[keep:], ends[keep:], strict=True):
                        shape, dtype, fmt = self._engine.layerwise_prefill_row_metadata(
                            group, end - start
                        )
                        obj = self._engine.storage_manager.allocate(
                            shape, dtype, fmt, busy_loop=False
                        )
                        if obj is None:
                            raise MemoryError(
                                "Layerwise-prefill shared CPU allocation failed"
                            )
                        with PinMonitor.GetOrCreate().protect_pins() as pins:
                            obj.pin()
                            pins.append(obj)
                        fresh.append(obj)
                        obj.metadata.cached_positions = torch.arange(
                            start, end, dtype=torch.long
                        )
                    self._transfer(
                        req, name, group, bank, fresh, starts[keep:], ends[keep:], True
                    )
                    # Strict put consumes its borrowed refs on every exit. The
                    # manifest keeps separate refs and request-lifetime pins.
                    for obj in fresh:
                        obj.ref_count_up()
                    self._engine.storage_manager.batched_put_sync_required(
                        keys[keep:],
                        fresh,
                        **(
                            {"location": "LocalCPUBackend"}
                            if self._page_first
                            else {"required_backends": self._required_backends}
                        ),
                    )
            except Exception as exc:
                error = exc
            # Even a root allocation/D2H/put failure must send an error envelope
            # to the passive peers waiting for publication, then join their ack.
            try:
                objects = self._engine.resolve_layerwise_prefill_row(
                    *identity,
                    keys[keep:],
                    starts[keep:],
                    ends[keep:],
                    phase=f"prefill_save:{self._step}",
                    memory_objs=fresh if root else None,
                    error=error,
                )
            except Exception as exc:
                error = error or exc
            try:
                self._ack(("save", identity), error)
            except Exception:
                acquired = fresh if root else (objects or [])
                if self._unsafe_transfer:
                    self._quarantined.extend(acquired)
                else:
                    self._release(acquired)
                raise
            self._release(prior.objects[keep:])
            self._prefixes[identity] = _RowPrefix(
                starts,
                ends,
                keys,
                prior.objects[:keep] + objects,
                revision=self._step,
                slab_generation=self._engine.shared_cpu_cache_generation,
            )
            self._published_handles += len(objects)
        self._ready.remove((group, row))
        self._saved[group] += 1
        self._save_seconds += perf_counter() - started
        return self._step_future

    def finish_step(self) -> None:
        """Validate all rows on all TP ranks, then commit complete CPU page groups.

        Page mode submits only changed chunks (the full prefix for a first
        external hit), waits required remote futures, and resolves the shared
        step future only after all ranks acknowledge commit. Any failure poisons
        the step and completes its future exceptionally without logging success.
        """
        error = None
        if (
            not self._bound
            or self._failed
            or self._ready
            or (self._requests and tuple(self._saved) != self._view.layer_counts)
        ):
            error = ValueError(
                f"Incomplete layerwise-prefill step: saved={self._saved}, "
                "expected=79/22"
            )
        if self._page_first and error is None:
            try:
                if self._pending_rows:
                    raise ValueError("Unpublished layerwise-prefill bootstrap rows")
                for req in self._requests:
                    keep = self._commit_starts[req.request_id] // 256
                    for group, count in enumerate(self._view.layer_counts):
                        plan = self._plans[req.request_id, group, req.compute_end]
                        prefixes = [
                            self._prefixes[
                                req.request_id, req.allocation_generation, group, row
                            ]
                            for row in range(count)
                        ]
                        for prefix in prefixes:
                            if (
                                not (
                                    len(prefix.starts)
                                    == len(prefix.ends)
                                    == len(prefix.keys)
                                    == len(prefix.objects)
                                    == len(plan)
                                )
                                or (prefix.starts and prefix.starts[0] != 0)
                                or (prefix.ends[-1] if prefix.ends else 0)
                                != req.compute_end
                                or prefix.revision != self._step
                                or prefix.slab_generation
                                != self._engine.shared_cpu_cache_generation
                            ):
                                raise ValueError("Invalid page commit manifest")
                        # Unchanged chunks retain trusted ownership and their
                        # keys were checked at save. Validate only the publish
                        # suffix; a first external hit deliberately starts at 0.
                        for chunk in range(keep, len(plan)):
                            start, end, key = plan[chunk]
                            shape, dtype, fmt = (
                                self._engine.layerwise_prefill_row_metadata(
                                    group, end - start
                                )
                            )
                            positions = torch.arange(start, end, dtype=torch.long)
                            for row, prefix in enumerate(prefixes):
                                if (
                                    prefix.starts[chunk] != start
                                    or prefix.ends[chunk] != end
                                    or prefix.keys[chunk] != key.get_layer(row)
                                ):
                                    raise ValueError("Invalid page commit manifest")
                                obj = prefix.objects[chunk]
                                if (
                                    not obj.is_valid()
                                    or obj.metadata.pin_count < 1
                                    or (
                                        obj.get_shape(),
                                        obj.get_dtype(),
                                        obj.get_memory_format(),
                                    )
                                    != (shape, dtype, fmt)
                                    or obj.metadata.cached_positions is None
                                    or not torch.equal(
                                        obj.metadata.cached_positions, positions
                                    )
                                ):
                                    raise ValueError(
                                        "Invalid page commit source metadata"
                                    )
            except Exception as exc:
                error = exc
        self._ack(("finish",), error)
        persist_started = perf_counter()
        pages = 0
        if self._page_first:
            error = None
            try:
                if self._engine.metadata.is_first_rank():
                    for req in self._requests:
                        keep = self._commit_starts[req.request_id] // 256
                        for group, count in enumerate(self._view.layer_counts):
                            plan = self._plans[req.request_id, group, req.compute_end]
                            for chunks in self._page_batches(req, group, keep):
                                keys, objects = [], []
                                for chunk in chunks:
                                    pages += plan[chunk][1] - plan[chunk][0] == 256
                                    for row in range(count):
                                        prefix = self._prefixes[
                                            req.request_id,
                                            req.allocation_generation,
                                            group,
                                            row,
                                        ]
                                        keys.append(prefix.keys[chunk])
                                        objects.append(prefix.objects[chunk])
                                for obj in objects:
                                    obj.ref_count_up()
                                self._submit_required_remote(keys, objects)
            except Exception as exc:
                error = exc
            try:
                self._drain_required_remote()
            except Exception as exc:
                error = error or exc
            self._ack(("commit",), error)
            self._step_future.set_result(None)
        persist_seconds = perf_counter() - persist_started
        # Host-observed wall times include synchronization waits, not isolated
        # NPU kernel time. Emit no per-layer records or diagnostic device reads.
        if self._engine.metadata.is_first_rank():
            elapsed = perf_counter() - self._step_started
            window_stats = getattr(self, "window_stats", None)
            logger.info(
                "[PREFILL_SYNC_STEP] event=end step=%d requests=%d saved=%s "
                "elapsed_ms=%.3f bind_ms=%.3f load_ms=%.3f save_ms=%.3f "
                "other_ms=%.3f reused_rows=%d published_handles=%d "
                "completed_ends=%s page_persist_ms=%.3f pages=%d "
                "prepared_slots=%d transfer_timing=%s "
                "bind_parts_ms=(plan,slots,ack):%s "
                "load_parts_ms=(prepare,submit,fence,ack):%s "
                "save_parts_ms=(prepare,submit,fence,ack):%s window_stats=%s",
                self._step,
                len(self._requests),
                tuple(self._saved),
                elapsed * 1000,
                self._bind_seconds * 1000,
                self._load_seconds * 1000,
                self._save_seconds * 1000,
                max(
                    0.0,
                    elapsed
                    - self._bind_seconds
                    - self._load_seconds
                    - self._save_seconds,
                )
                * 1000,
                self._reused_rows,
                self._published_handles,
                [(req.request_id, req.compute_end) for req in self._requests],
                persist_seconds * 1000 if self._page_first else 0.0,
                pages,
                self._prepared_slot_count,
                self._transfer_timing_enabled,
                tuple(round(value * 1000, 3) for value in self._bind_parts),
                *(
                    tuple(
                        round(timing.get(key, 0.0) * 1000, 3)
                        for key in ("prepare_s", "submit_s", "fence_s")
                    )
                    + (round(self._ack_seconds.get(phase, 0.0) * 1000, 3),)
                    for phase, timing in zip(
                        ("load", "save"), self._transfer_timings, strict=True
                    )
                ),
                window_stats() if callable(window_stats) else {},
            )
        self._bound = False
        self._requests = ()
        self._slots.clear()
        self._plans.clear()
        self._commit_starts.clear()

    def abort_request(self, request_id: str) -> None:
        """Release a completed or aborted request's retained row references/pins."""
        if any(req.request_id == request_id for req in self._requests):
            if self._step_future is not None and not self._step_future.done():
                self._step_future.set_exception(
                    ValueError("Layerwise-prefill step aborted")
                )
        if self._unsafe_transfer:
            raise LayerwisePrefillFenceError(
                "Cannot release shared CPU rows after a failed device fence; "
                "restart the worker"
            )
        req = self._history.pop(request_id, None)
        if req is not None:
            self._released[request_id] = max(
                req.allocation_generation, self._released.get(request_id, 0)
            )
        for identity in tuple(self._prefixes):
            if identity[0] == request_id:
                self._release(self._prefixes.pop(identity).objects)
        for identity in tuple(self._pending_rows):
            if identity[0] == request_id:
                self._release(self._pending_rows.pop(identity))
        if any(req.request_id == request_id for req in self._requests):
            self._failed = True
        if not self._history:
            self._requests = ()
            self._slots.clear()
            self._plans.clear()
            self._commit_starts.clear()
            self._bound = self._failed = False

    def _validate(
        self,
        metadata: Any,
        phase: str,
        kv_layer: Any = None,
        *,
        validation_error: Exception | None = None,
    ) -> tuple[int, int, int, str]:
        error = None
        key = None
        manifests = []
        ready = False
        try:
            if validation_error is not None:
                raise validation_error
            if not self._bound or self._failed or not self._requests:
                raise ValueError("Layerwise-prefill callback has no live bound step")
            row = metadata.row
            key = (
                row.layer_name,
                row.execution_ordinal,
                row.kv_group,
                row.row_ordinal,
                row.bank,
            )
            group, ordinal, bank = key[2:]
            if (
                any(type(value) is not int for value in key[1:])
                or group not in (0, 1)
                or not 0 <= ordinal < self.layer_count(group)
                or self._view.rows_by_group[group][ordinal] != key
            ):
                raise ValueError(
                    "Layerwise-prefill callback row differs from the frozen topology"
                )
            execution = metadata.execution
            actual_execution = tuple(
                None
                if member is None
                else (
                    member.layer_name,
                    member.execution_ordinal,
                    member.kv_group,
                    member.row_ordinal,
                    member.bank,
                )
                for member in (execution.latent, execution.indexer)
            )
            if (
                execution.execution_ordinal,
                *actual_execution,
            ) != self._view.executions[key[1]]:
                raise ValueError(
                    "Layerwise-prefill execution differs from the frozen topology"
                )
            expected = tuple(
                (req.request_id, req.allocation_generation) for req in self._requests
            )
            if metadata.request_generations != expected or any(
                type(generation) is not int
                for _, generation in metadata.request_generations
            ):
                raise ValueError(
                    "Layerwise-prefill callback differs from exact request bindings"
                )
            if ordinal != self._saved[group]:
                raise ValueError("Layerwise-prefill callback is out of group row order")
            ready = (group, ordinal) in self._ready
            # Agree on ownership before choosing bootstrap vs. warm reuse. The
            # exact bind plans plus this committed summary also agree on keep.
            for req in self._requests:
                prior = self._prefixes.get(
                    (req.request_id, req.allocation_generation, group, ordinal)
                )
                if prior is None:
                    manifests.append((False, None, None, None, None))
                    if phase == "save" or ready:
                        raise ValueError("Missing retained layerwise-prefill manifest")
                    continue
                extent = prior.ends[-1] if prior.ends else 0
                count = len(prior.objects)
                manifests.append(
                    (True, prior.revision, extent, count, prior.slab_generation)
                )
                if (
                    prior.slab_generation != self._engine.shared_cpu_cache_generation
                    or not 0 < prior.revision <= self._step
                    or extent != req.restore_end
                    or count != (extent + 255) // 256
                    or not (
                        len(prior.starts) == len(prior.ends) == len(prior.keys) == count
                    )
                ):
                    raise ValueError("Stale layerwise-prefill manifest")
            if phase == "save":
                if not ready:
                    raise ValueError(
                        "Layerwise-prefill save arrived before row restore"
                    )
                planes = self._planes(kv_layer, group)
                for actual, bound in zip(planes, self._caches[key[0]], strict=True):
                    if (
                        actual.data_ptr(),
                        actual.shape,
                        actual.stride(),
                        actual.device,
                    ) != (bound.data_ptr(), bound.shape, bound.stride(), bound.device):
                        raise ValueError(
                            "Layerwise-prefill KV planes differ from the bound registry"
                        )
        except Exception as exc:
            error = exc
        self._ack(("validate", phase, key, ready, tuple(manifests)), error)
        return group, ordinal, bank, key[0]

    def _ack(self, identity: tuple, error: Exception | None = None) -> None:
        started = perf_counter()
        phase = identity[1] if identity[0] == "validate" else identity[0]
        try:
            self._engine.layerwise_prefill_ack((self._step, identity), error)
        except Exception as exc:
            self._failed = True
            if isinstance(exc, LayerwisePrefillFenceError):
                self._unsafe_transfer = True
            if self._step_future is not None and not self._step_future.done():
                self._step_future.set_exception(exc)
            for objects in self._pending_rows.values():
                if self._unsafe_transfer:
                    self._quarantined.extend(objects)
                else:
                    self._release(objects)
            self._pending_rows.clear()
            raise
        finally:
            self._ack_seconds[phase] = (
                self._ack_seconds.get(phase, 0.0) + perf_counter() - started
            )

    def _transfer(
        self,
        req: LayerwisePrefillRequest,
        name: str,
        group: int,
        bank: int,
        objects: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        direction: bool,
    ) -> None:
        connector = self._engine.gpu_connector
        try:
            connector.transfer_layerwise_prefill_row(
                self._caches[name],
                [obj.tensor for obj in objects],
                starts,
                ends,
                self._slots[req.request_id, bank, group],
                kv_group=group,
                direction=direction,
                **(
                    {"timing": self._transfer_timings[int(direction)]}
                    if self._transfer_timing_enabled
                    else {}
                ),
            )
        except Exception:
            # A retained Tensor alone does not protect an allocator's shm offset
            # from reuse. If even an explicit error-path fence fails, quarantine
            # MemoryObj ownership on ALL ranks, including the rank0 slab owner.
            try:
                connector.synchronize_dense_load_stream()
                connector.synchronize_shared_cpu_store_publication()
            except Exception as exc:
                raise LayerwisePrefillFenceError(
                    "Layerwise-prefill device completion fence failed"
                ) from exc
            raise

    def _page_batches(
        self, req: LayerwisePrefillRequest, group: int, keep: int
    ) -> Iterator[range]:
        """Yield indivisible token chunks for strict complete-group publication."""
        plan = self._plans[req.request_id, group, req.compute_end]
        for base in range(keep, len(plan), 16):
            yield range(base, min(base + 16, len(plan)))

    def _submit_required_remote(self, keys: list, objects: list[MemoryObj]) -> None:
        """Consume borrowed references through required remote persistence."""
        self._engine.storage_manager.batched_put_sync_required(
            keys,
            objects,
            required_backends=("RemoteBackend",),
            location="RemoteBackend",
        )

    def _drain_required_remote(self) -> None:
        """Join any deferred strict puts before the final all-TP commit ACK."""

    def _plan(
        self, req: LayerwisePrefillRequest, end: int, group: int, row: int
    ) -> tuple[list[int], list[int], list[CacheEngineKey]]:
        plan = self._plans[req.request_id, group, end]
        prior = self._prefixes.get(
            (req.request_id, req.allocation_generation, group, row)
        )
        keep = (
            min(req.compute_start, prior.ends[-1] if prior.ends else 0) // 256
            if prior is not None and end == req.compute_end
            else 0
        )
        starts, ends, keys = [], [], []
        for chunk, (start, stop, key) in enumerate(plan):
            starts.append(start)
            ends.append(stop)
            retained = prior.keys[chunk] if chunk < keep else None
            # Match the entire get_layer projection, including metadata ignored
            # by key equality. Subclasses/custom keys keep their original API.
            if (
                type(key) is CacheEngineKey
                and type(retained) is LayerCacheEngineKey
                and prior.starts[chunk] == start
                and prior.ends[chunk] == stop
                and row == retained.layer_id
                and key.model_name == retained.model_name
                and key.world_size == retained.world_size
                and key.worker_id == retained.worker_id
                and key.chunk_hash == retained.chunk_hash
                and key.dtype == retained.dtype
                and key.request_configs == retained.request_configs
                and key.tags == retained.tags
                and key._dtype_str == retained._dtype_str
                and key.kv_group == retained.kv_group
            ):
                keys.append(retained)
            else:
                keys.append(key.get_layer(row))
        return starts, ends, keys

    @staticmethod
    def _planes(value: Any, group: int) -> tuple[torch.Tensor, ...]:
        # The registered-layer ABI uses tuples, including singleton INDEXER
        # planes. Preserve tensor/storage identity when normalizing inputs.
        planes = tuple(value) if isinstance(value, (list, tuple)) else (value,)
        if len(planes) != (2 if group == 0 else 1) or any(
            not isinstance(plane, torch.Tensor)
            or plane.dtype != torch.bfloat16
            or plane.ndim != 4
            or not plane.is_contiguous()
            for plane in planes
        ):
            raise ValueError("Layerwise-prefill requires BF16 contiguous NHD KV planes")
        if any(
            plane.shape[:2] != planes[0].shape[:2] or plane.device != planes[0].device
            for plane in planes
        ):
            raise ValueError("Layerwise-prefill KV plane block layouts differ")
        return planes

    @staticmethod
    def _release(objects: list[MemoryObj]) -> None:
        for obj in objects:
            PinMonitor.GetOrCreate().release_pin_lease(obj)
            obj.ref_count_down()
