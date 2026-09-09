# SPDX-License-Identifier: Apache-2.0
"""Step-bounded shared-bank transfers with end-of-step required page persistence.

Only the model thread prepares, enqueues, fences and publishes rows. Storage
workers see ready immutable CPU pages, never device tickets or TP collectives.
The shared step Future is an assembly/commit barrier, not an admission credit.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, NoReturn

# Third Party
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.required_put_queue import RequiredPutQueue
import torch

# First Party
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillFenceError,
    LayerwisePrefillSyncBackend,
    _RowPrefix,
)


@dataclass
class _SaveSource:
    identity: tuple
    prior: _RowPrefix
    starts: list[int]
    ends: list[int]
    keys: list
    keep: int
    fresh: list[MemoryObj] = field(default_factory=list)


@dataclass
class _AsyncRow:
    # Includes step, job ID, canonical row/bank and exact request ranges. No
    # completion callback ever resolves ownership using request ID alone.
    identity: tuple
    key: tuple
    loads: list[Any] = field(default_factory=list)
    stores: list[Any] = field(default_factory=list)
    sources: list[_SaveSource] = field(default_factory=list)
    load_submitted: bool = False
    save_prepared: bool = False
    save_submitted: bool = False
    consumed: bool = False
    finished: bool = False
    bytes: int = 0


class LayerwisePrefillAsyncBackend(LayerwisePrefillSyncBackend):
    """Actual async row backend; all device and storage work ends within a step.

    Call configure_window_limits on all ranks before bind_step. Row entry calls
    wait_for_load, then post-SFA/pre-HCOM calls submit_save and submit_load.
    finish_save is unconditional post-HCOM, including coordinator validation
    errors. abort_step must be called on every rank on model exceptions.
    Unknown fences permanently poison this instance and retain slab ownership.
    """

    supports_transfer_window = True
    manages_pending_work = True
    accepts_coordinator_validation_errors = True

    def __init__(self, engine: Any, view: Any) -> None:
        super().__init__(engine, view)
        # Async prepare/submit/complete span different callbacks; the synchronous
        # row-hook timing buckets do not measure this path.
        self._transfer_timing_enabled = False
        self._limits = (8, 64 << 20, 8)
        self._rows: dict[tuple[int, int], _AsyncRow] = {}
        self._bank_events: dict[tuple[int, int], Any] = {}
        self._load_executions: set[int] = set()
        self._pending_error: Exception | None = None
        self._poisoned = False
        self._abort_drained = False
        self._failure_error: Exception | None = None
        self._callbacks: dict[tuple[int, int], Any] = {}
        self._previous_callbacks: tuple[Any, ...] = ()
        self._queue: RequiredPutQueue | None = None
        self._job_id = 0
        self._device_jobs = self._remote_jobs = 0
        self._peak_jobs = self._peak_bytes = self._peak_futures = 0

    def configure_window_limits(
        self, max_jobs: int, max_bytes: int, max_futures: int
    ) -> None:
        """Set positive integer budgets without constructing a StorageManager.

        Configuration errors are agreed at bind, before transfers or storage I/O.
        Device jobs count row batches, not requests; futures count remote I/O only.
        """
        if self._bound:
            raise ValueError("Cannot change transfer limits during a bound step")
        self._limits = (max_jobs, max_bytes, max_futures)

    def bind_step(
        self,
        requests: list[LayerwisePrefillRequest],
        kv_caches: dict[str, Any],
        *,
        callbacks: tuple[Any, ...] | None = None,
        validation_error: Exception | None = None,
    ) -> None:
        """Bind all 101 actual forward callback objects, plans, slots and limits.

        Callbacks are immutable per-forward submission tokens. Aliases may repeat
        the same object, never a different object for the same canonical row.
        Missing callbacks or reuse from the preceding forward fail before I/O.
        TP agrees canonical values, not process-local object IDs.
        """
        if self._poisoned and (self._abort_drained or self._unsafe_transfer):
            failure_type = (
                LayerwisePrefillFenceError if self._unsafe_transfer else ValueError
            )
            raise failure_type(
                "Async backend is poisoned; restart worker"
            ) from self._failure_error
        error = validation_error
        registered = {}
        generations = ()
        try:
            if error is not None:
                raise error
            if self._unsafe_transfer:
                raise LayerwisePrefillFenceError(
                    "Async backend is poisoned; restart worker"
                )
            if self._poisoned:
                raise ValueError("Async backend is poisoned; restart worker")
            if self._bound:
                raise ValueError("Previous async step is unfinished")
            if not isinstance(callbacks, tuple):
                raise ValueError(
                    "Async bind requires the complete forward callbacks tuple"
                )
            generations = tuple(
                (r.request_id, r.allocation_generation) for r in requests
            )
            previous_ids = {
                id(c) for c in (*self._previous_callbacks, *self._callbacks.values())
            }
            for callback in callbacks:
                key = self._callback_key(callback, generations)
                row = key[2:4]
                if id(callback) in previous_ids:
                    raise ValueError(
                        "Async callback object reused from a previous step"
                    )
                if row in registered and registered[row] is not callback:
                    raise ValueError("Different callback objects alias the same row")
                registered[row] = callback
            if len(registered) != sum(self._view.layer_counts):
                raise ValueError("Async bind requires all 101 canonical row callbacks")
            if not self._page_first:
                raise ValueError("Async layerwise prefill requires page-first storage")
            connector = self._engine.gpu_connector
            if (
                getattr(connector, "supports_layerwise_prefill_async_rows", False)
                is not True
            ):
                raise ValueError("Connector does not support async prefill rows")
            for name in (
                "prepare_layerwise_prefill_row",
                "submit_layerwise_prefill_row",
                "wait_layerwise_prefill_row",
                "complete_layerwise_prefill_row",
                "drain_layerwise_prefill_transfers",
            ):
                if not callable(getattr(connector, name, None)):
                    raise ValueError(f"Missing async connector hook: {name}")
            if any(type(value) is not int or value <= 0 for value in self._limits):
                raise ValueError("Transfer window limits must be positive integers")
            manager = getattr(self._engine, "storage_manager", None)
            if (
                self._engine.metadata.is_first_rank()
                and manager is not None
                and type(manager.storage_backends.get("RemoteBackend"))
                is not RemoteBackend
            ):
                raise ValueError("Async prefill requires the standard RemoteBackend")
            jobs, budget, _ = self._limits
            row_bytes = []
            for group, rows in enumerate(self._view.rows_by_group):
                planes = self._planes(kv_caches[rows[0][0]], group)
                width = sum(p.shape[2] * p.shape[3] * p.element_size() for p in planes)
                if width * 256 * len(rows) > budget:
                    raise ValueError("Indivisible full page exceeds max_bytes")
                row_bytes.append(
                    sum(
                        width
                        * (
                            req.compute_end
                            - min(req.compute_start, req.restore_end) // 256 * 256
                        )
                        for req in requests
                    )
                )
            for _, latent, indexer in self._view.executions:
                groups = [
                    g for g, row in enumerate((latent, indexer)) if row is not None
                ]
                if requests and (
                    len(groups) > jobs or sum(row_bytes[g] for g in groups) > budget
                ):
                    raise ValueError(
                        "Current execution row batches exceed transfer budget"
                    )
        except Exception as exc:
            error = exc
        # This ACK also freezes identical limits on passive ranks, which have no
        # storage manager. Do not prepare/upload even common slots on rejection.
        self._ack(
            (
                "window_bind",
                self._step + 1,
                self._limits,
                generations,
                tuple(self._view.rows_by_group[g][r] for g, r in sorted(registered)),
            ),
            error,
        )
        try:
            super().bind_step(requests, kv_caches, validation_error=error)
        except Exception as exc:
            # The sync bind already performed its collective; enter the same
            # failure handshake without issuing another initial phase ACK.
            self._fail_ack(exc)
        self._step_future.set_running_or_notify_cancel()
        self._rows.clear()
        self._bank_events.clear()
        self._load_executions.clear()
        self._pending_error = None
        self._previous_callbacks = tuple(self._callbacks.values())
        self._callbacks = registered
        self._abort_drained = False
        self._queue = None
        self._device_jobs = self._remote_jobs = 0
        self._peak_jobs = self._peak_bytes = self._peak_futures = 0

    def pending_jobs(self) -> int:
        """Return SOURCE_DONE-pending row batches plus ready remote jobs."""
        return sum(r.save_submitted and not r.finished for r in self._rows.values()) + (
            self._queue.pending_jobs if self._queue is not None else 0
        )

    def pending_bytes(self) -> int:
        """Return admitted D2H payload and ready-remote logical bytes, not manifests."""
        return sum(
            r.bytes for r in self._rows.values() if r.save_submitted and not r.finished
        ) + (self._queue.pending_bytes if self._queue is not None else 0)

    def pending_futures(self) -> int:
        """Return pending remote child futures; the assembly Future costs no slot."""
        return self._queue.pending_futures if self._queue is not None else 0

    def window_stats(self) -> dict[str, int]:
        """Return this step's limits, peaks, pending counts and actual job counts."""
        stats = self._queue.stats() if self._queue is not None else {}
        return {
            "max_jobs": self._limits[0],
            "max_bytes": self._limits[1],
            "max_futures": self._limits[2],
            "pending_jobs": self.pending_jobs(),
            "pending_bytes": self.pending_bytes(),
            "pending_futures": self.pending_futures(),
            "peak_jobs": max(self._peak_jobs, stats.get("peak_jobs", 0)),
            "peak_bytes": max(self._peak_bytes, stats.get("peak_bytes", 0)),
            "peak_futures": max(self._peak_futures, stats.get("peak_futures", 0)),
            "actual_jobs": self._device_jobs + self._remote_jobs,
            "device_jobs": self._device_jobs,
            "remote_jobs": self._remote_jobs,
        }

    def wait_for_load(
        self, metadata: Any, *, validation_error: Exception | None = None
    ) -> None:
        """Agree current readiness, prepare current save and next group-local load.

        CPU source resolution and all metadata/allocation work occur here, before
        SFA. Already submitted loads only add compute-event waits, not host fences.
        Lookahead is not marked consumed/ready until its real row enters.
        """
        started = perf_counter()
        try:
            self._registered_callback(metadata)
        except Exception as exc:
            validation_error = validation_error or exc
        group, row, bank, name = self._validate(
            metadata, "load", validation_error=validation_error or self._pending_error
        )
        current = self._prepare_load(group, row)
        error = None
        try:
            if current.consumed:
                raise ValueError("Duplicate async row entry")
            self._prepare_save(current)
        except Exception as exc:
            error = exc
        self._prepared_ack(("prepare_save", group, row), error)
        if row + 1 < self.layer_count(group):
            self._prepare_load(group, row + 1)
        error = None
        try:
            if not current.load_submitted:
                self._enqueue_load(current)
            for ticket in current.loads:
                self._engine.gpu_connector.wait_layerwise_prefill_row(ticket)
        except Exception as exc:
            error = exc
        self._prepared_ack(("load_ready", current.identity), error)
        current.consumed = True
        self._ready.add((group, row))
        self._load_seconds += perf_counter() - started

    def submit_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
        *,
        validation_error: Exception | None = None,
    ) -> None:
        """Pre-HCOM enqueue only; defer every ordinary error to finish_save."""
        try:
            record = self._local_row(metadata, validation_error)
            if not record.consumed or not record.save_prepared or record.save_submitted:
                raise ValueError("Async save is unprepared or duplicated")
            group, row, bank = record.key[2:]
            planes = self._planes(kv_layer, group)
            for actual, bound in zip(planes, self._caches[record.key[0]], strict=True):
                if (
                    actual.data_ptr(),
                    actual.shape,
                    actual.stride(),
                    actual.device,
                ) != (bound.data_ptr(), bound.shape, bound.stride(), bound.device):
                    raise ValueError("Async KV planes differ from bound registry")
            # Register the batch before any enqueue can throw. The pre-existing
            # record already owns every fresh allocation and prepared ticket.
            record.save_submitted = True
            self._device_jobs += 1
            self._peak_jobs = max(self._peak_jobs, self.pending_jobs())
            self._peak_bytes = max(self._peak_bytes, self.pending_bytes())
            for ticket in record.stores:
                self._engine.gpu_connector.submit_layerwise_prefill_row(ticket)
            self._bank_events[group, bank] = record.stores[-1].done_event
        except Exception as exc:
            self._pending_error = self._pending_error or exc

    def submit_load(
        self, metadata: Any, *, validation_error: Exception | None = None
    ) -> None:
        """Pre-HCOM enqueue prepared successors for actual execution groups only."""
        try:
            current = self._local_row(metadata, validation_error)
            execution = current.key[1]
            if execution in self._load_executions:
                raise ValueError("Duplicate async lookahead submission")
            self._load_executions.add(execution)
            if self._pending_error is not None:
                return
            for key in self._view.executions[execution][1:]:
                if key is None:
                    continue
                group, row = key[2:4]
                source = self._rows[group, row]
                if not source.save_submitted:
                    raise ValueError("Async lookahead arrived before current save")
                if row + 1 < self.layer_count(group):
                    self._enqueue_load(self._rows[group, row + 1])
        except Exception as exc:
            self._pending_error = self._pending_error or exc

    def finish_save(
        self, metadata: Any, *, validation_error: Exception | None = None
    ) -> Future:
        """Unconditionally agree post-HCOM, fence sources, then publish/adopt rows.

        SOURCE_DONE frees row admission credits. The returned shared Future stays
        unfinished until finish_step has committed all pages and all ranks ACK.
        """
        started = perf_counter()
        error = self._pending_error
        record = None
        try:
            record = self._local_row(metadata, validation_error)
            if not record.save_submitted or record.finished:
                raise ValueError("Async finish is missing its submit or duplicated")
            if error is None:
                for ticket in (*record.loads, *record.stores):
                    self._engine.gpu_connector.complete_layerwise_prefill_row(ticket)
        except Exception as exc:
            error = error or exc
        if error is not None:
            drained = self._drain_devices()
            if isinstance(drained, LayerwisePrefillFenceError):
                error = drained
        self._ack(("source_done", None if record is None else record.identity), error)
        root = self._engine.metadata.is_first_rank()
        for source in record.sources:
            error = None
            objects = None
            keep = source.keep
            try:
                if root:
                    for obj in source.fresh:
                        obj.ref_count_up()
                    self._engine.storage_manager.batched_put_sync_required(
                        source.keys[keep:], source.fresh, location="LocalCPUBackend"
                    )
            except BaseException as exc:
                error = self._storage_error(exc)
            try:
                objects = self._engine.resolve_layerwise_prefill_row(
                    *source.identity,
                    source.keys[keep:],
                    source.starts[keep:],
                    source.ends[keep:],
                    phase=f"prefill_save:{self._step}",
                    memory_objs=source.fresh if root else None,
                    error=error,
                )
                if not root:
                    source.fresh = objects
            except Exception as exc:
                error = error or exc
            self._ack(("save", record.identity, source.identity), error)
            self._release(source.prior.objects[keep:])
            self._prefixes[source.identity] = _RowPrefix(
                source.starts,
                source.ends,
                source.keys,
                source.prior.objects[:keep] + objects,
                self._step,
                self._engine.shared_cpu_cache_generation,
            )
            source.fresh = []  # Ownership moved into the committed manifest.
            self._published_handles += len(objects)
        record.finished = True
        record.loads.clear()
        record.stores.clear()
        record.sources.clear()
        group, row = record.key[2:4]
        self._ready.remove((group, row))
        self._saved[group] += 1
        self._save_seconds += perf_counter() - started
        return self._step_future

    def sync_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
        *,
        validation_error: Exception | None = None,
    ) -> Future:
        """Compatibility entry using the prepared async row, never legacy transfer."""
        self.submit_save(
            metadata, kv_layer, attn_metadata, validation_error=validation_error
        )
        return self.finish_save(metadata, validation_error=validation_error)

    def finish_step(self) -> None:
        """Drain every device operation first, then validate and commit 79/22 rows."""
        error = self._drain_devices()
        if error is None:
            error = self._pending_error
        self._ack(("device_finish",), error)
        super().finish_step()
        self._rows.clear()
        self._bank_events.clear()

    def abort_step(self) -> None:
        """Drain, poison and retire the entire active batch on every TP rank.

        An abort may meet a peer's normal phase ACK. Both ranks then enter the
        common failure-drain handshake; a later cleanup never repeats that ACK.
        A mismatched/failed ACK is re-raised after safe cleanup, not swallowed.
        Unknown fences quarantine ALL owners and forbid further collectives.
        """
        if self._unsafe_transfer:
            raise LayerwisePrefillFenceError(
                "Cannot release async slab owners; restart worker"
            ) from self._failure_error
        if not self._bound:
            return
        self._poisoned = True
        failure = None
        if not self._abort_drained:
            error = self._drain_devices()
            try:
                self._drain_required_remote()
            except Exception as exc:
                error = error or exc
            try:
                self._ack(("abort_devices",), error)
            except Exception as exc:
                failure = exc
            else:
                self._abort_drained = True
        if self._unsafe_transfer:
            raise LayerwisePrefillFenceError(
                "Cannot release async slab owners; restart worker"
            ) from failure
        self._failure_error = (
            self._failure_error or failure or ValueError("Async prefill step aborted")
        )
        if self._step_future is not None and not self._step_future.done():
            self._step_future.set_exception(self._failure_error)
        for record in self._rows.values():
            for source in record.sources:
                self._release(source.fresh)
                source.fresh = []
            record.loads.clear()
            record.stores.clear()
        self._rows.clear()
        self._bank_events.clear()
        for req in self._requests:
            super().abort_request(req.request_id)
        self._requests = ()
        self._slots.clear()
        self._plans.clear()
        self._commit_starts.clear()
        self._ready.clear()
        self._bound = False
        self._failed = True
        if failure is not None:
            raise failure

    def abort_request(
        self, request_id: str, *, allocation_generation: int | None = None
    ) -> None:
        """Retire only matching request ownership; stale generation cleanup is a no-op.

        Retiring a completed request does not affect another bound batch, its
        device work or Future, and uses no collective. Cancelling a participant
        ends the entire active batch through abort_step; partial continuation is
        unsupported. Unknown fences retain all matching owners until restart.
        """
        req = self._history.get(request_id)
        if req is None or (
            allocation_generation is not None
            and (
                type(allocation_generation) is not int
                or allocation_generation != req.allocation_generation
            )
        ):
            return
        if any(r.request_id == request_id for r in self._requests):
            self.abort_step()
        else:
            super().abort_request(request_id)

    def _local_row(self, metadata: Any, error: Exception | None = None) -> _AsyncRow:
        if error is not None:
            raise error
        if not self._bound or self._failed or not self._requests:
            raise ValueError("Async callback has no live step")
        key = self._registered_callback(metadata)
        group, row = key[2:4]
        record = self._rows[group, row]
        if record.key != key or record.identity[0] != self._step:
            raise ValueError("Async callback differs from prepared row")
        if row != self._saved[group]:
            raise ValueError("Async callback is stale or out of row order")
        return record

    def _registered_callback(self, metadata: Any) -> tuple:
        key = self._callback_key(
            metadata,
            tuple((r.request_id, r.allocation_generation) for r in self._requests),
        )
        if self._callbacks.get(key[2:4]) is not metadata:
            raise ValueError("Async callback object is not registered for this step")
        return key

    def _callback_key(self, metadata: Any, generations: tuple) -> tuple:
        key = self._row_key(metadata.row)
        if key is None:
            raise ValueError("Async callback has no canonical row")
        group, row = key[2:4]
        if (
            group not in (0, 1)
            or not 0 <= row < self.layer_count(group)
            or self._view.rows_by_group[group][row] != key
        ):
            raise ValueError("Async callback differs from canonical row membership")
        if metadata.request_generations != generations or any(
            type(g) is not int for _, g in metadata.request_generations
        ):
            raise ValueError("Async callback differs from exact request generations")
        execution = metadata.execution
        if (
            type(execution.execution_ordinal) is not int
            or (
                execution.execution_ordinal,
                self._row_key(execution.latent),
                self._row_key(execution.indexer),
            )
            != self._view.executions[key[1]]
        ):
            raise ValueError("Async callback differs from frozen execution")
        return key

    @staticmethod
    def _row_key(row: Any) -> tuple | None:
        if row is None:
            return None
        key = (
            row.layer_name,
            row.execution_ordinal,
            row.kv_group,
            row.row_ordinal,
            row.bank,
        )
        if any(type(v) is not int for v in key[1:]):
            raise ValueError("Invalid async row identity")
        return key

    def _prepare_load(self, group: int, row: int) -> _AsyncRow:
        if (group, row) in self._rows:
            return self._rows[group, row]
        key = self._view.rows_by_group[group][row]
        self._job_id += 1
        record = _AsyncRow(
            (
                self._step,
                self._job_id,
                key,
                tuple(
                    (
                        r.request_id,
                        r.allocation_generation,
                        r.compute_start,
                        r.compute_end,
                        r.restore_end,
                    )
                    for r in self._requests
                ),
            ),
            key,
        )
        self._rows[group, row] = record
        for req in self._requests:
            identity = (req.request_id, req.allocation_generation, group, row)
            prior = self._prefixes.get(identity)
            error = None
            starts, ends, keys = [], [], []
            summary = (False, None, None, None, None)
            try:
                if prior is None:
                    starts, ends, keys = self._plan(req, req.restore_end, group, row)
                else:
                    starts, ends, keys = prior.starts, prior.ends, prior.keys
                    extent, count = ends[-1] if ends else 0, len(prior.objects)
                    summary = (
                        True,
                        prior.revision,
                        extent,
                        count,
                        prior.slab_generation,
                    )
                    if (
                        extent != req.restore_end
                        or count != (extent + 255) // 256
                        or not len(starts) == len(ends) == len(keys) == count
                        or prior.slab_generation
                        != self._engine.shared_cpu_cache_generation
                        or not 0 < prior.revision <= self._step
                    ):
                        raise ValueError("Stale async lookahead manifest")
            except Exception as exc:
                error = exc
            # Bind agrees exact plans; retained manifests already own validated
            # keys and shapes. Do not rebuild/broadcast long warm key lists.
            self._ack(("prepare_load", identity, summary), error)
            objects = None
            try:
                if prior is None:
                    if self._engine.metadata.is_first_rank():
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
                            objects = self._pending_rows[identity]
                        except BaseException as exc:
                            error = self._storage_error(exc)
                    objects = self._engine.resolve_layerwise_prefill_row(
                        *identity,
                        keys,
                        starts,
                        ends,
                        phase=f"prefill_load:{self._step}",
                        memory_objs=objects,
                        error=error,
                    )
                    # Install ownership before device preparation can enqueue.
                    self._pending_rows.pop(identity, None)
                    prior = _RowPrefix(
                        starts,
                        ends,
                        keys,
                        objects,
                        self._step,
                        self._engine.shared_cpu_cache_generation,
                    )
                    self._prefixes[identity] = prior
                    self._published_handles += len(keys)
                else:
                    self._reused_rows += 1
                record.loads.append(
                    self._prepare_ticket(req, key, prior.objects, starts, ends, False)
                )
            except Exception as exc:
                error = error or exc
            self._prepared_ack(("load_prepared", record.identity, identity), error)
        return record

    def _prepare_save(self, record: _AsyncRow) -> None:
        root = self._engine.metadata.is_first_rank()
        group, row = record.key[2:4]
        for req in self._requests:
            identity = (req.request_id, req.allocation_generation, group, row)
            prior = self._prefixes[identity]
            starts, ends, keys = self._plan(req, req.compute_end, group, row)
            keep = min(req.compute_start, prior.ends[-1] if prior.ends else 0) // 256
            source = _SaveSource(identity, prior, starts, ends, keys, keep)
            record.sources.append(source)
            if keys[:keep] != prior.keys[:keep]:
                raise ValueError("Async unchanged chunk keys differ")
            if root:
                for start, end in zip(starts[keep:], ends[keep:], strict=True):
                    shape, dtype, fmt = self._engine.layerwise_prefill_row_metadata(
                        group, end - start
                    )
                    try:
                        obj = self._engine.storage_manager.allocate(
                            shape, dtype, fmt, busy_loop=False
                        )
                    except BaseException as exc:
                        raise self._storage_error(exc) from exc
                    if obj is None:
                        raise MemoryError("Async shared CPU allocation failed")
                    with PinMonitor.GetOrCreate().protect_pins() as pins:
                        obj.pin()
                        pins.append(obj)
                    source.fresh.append(obj)
                    record.bytes += obj.get_size()
                    obj.metadata.cached_positions = torch.arange(
                        start, end, dtype=torch.long
                    )
                record.stores.append(
                    self._prepare_ticket(
                        req, record.key, source.fresh, starts[keep:], ends[keep:], True
                    )
                )
        if not root:
            # Passive ranks still record the post-SFA dependency for bank reuse.
            record.stores.append(
                self._prepare_ticket(self._requests[0], record.key, [], [], [], True)
            )
        if (
            record.bytes
            + sum(
                r.bytes
                for r in self._rows.values()
                if r is not record and r.save_prepared and not r.finished
            )
            > self._limits[1]
        ):
            raise ValueError("Actual current row payload exceeds max_bytes")
        record.save_prepared = True

    def _prepare_ticket(
        self,
        req: Any,
        key: tuple,
        objects: list,
        starts: list,
        ends: list,
        direction: bool,
    ) -> Any:
        return self._engine.gpu_connector.prepare_layerwise_prefill_row(
            self._caches[key[0]],
            [obj.tensor for obj in objects],
            starts,
            ends,
            self._slots[req.request_id, key[4], key[2]],
            kv_group=key[2],
            direction=direction,
        )

    def _enqueue_load(self, record: _AsyncRow) -> None:
        if record.load_submitted:
            raise ValueError("Duplicate async load submission")
        record.load_submitted = True
        event = self._bank_events.get((record.key[2], record.key[4]))
        for ticket in record.loads:
            self._engine.gpu_connector.submit_layerwise_prefill_row(
                ticket, wait_event=event
            )

    def _prepared_ack(self, identity: tuple, error: Exception | None) -> None:
        if error is not None:
            drained = self._drain_devices()
            if isinstance(drained, LayerwisePrefillFenceError):
                error = drained
        self._ack(identity, error)

    def _drain_devices(self) -> Exception | None:
        connector = self._engine.gpu_connector
        error = unknown = None
        try:
            connector.drain_layerwise_prefill_transfers()
        except Exception as exc:
            # Drain rethrows even a safely fenced native submit error. A
            # second drain distinguishes that from sticky unknown-fence poison.
            error = exc
            try:
                connector.drain_layerwise_prefill_transfers()
            except Exception as exc:
                unknown = exc
        # Successful preparation queues metadata without an inflight submission.
        # Always attempt BOTH streams, even when an earlier completion failed.
        for name in (
            "synchronize_dense_load_stream",
            "synchronize_shared_cpu_store_publication",
        ):
            try:
                getattr(connector, name)()
            except Exception as exc:
                unknown = unknown or exc
        if unknown is not None:
            self._unsafe_transfer = self._poisoned = True
            return LayerwisePrefillFenceError(
                f"Async device completion is unknown: {unknown}"
            )
        return error

    def _ack(self, identity: tuple, error: Exception | None = None) -> None:
        if self._failure_error is not None:
            raise self._failure_error
        started = perf_counter()
        try:
            self._engine.layerwise_prefill_ack((self._step, identity), error)
        except Exception as exc:
            self._fail_ack(exc)
        finally:
            phase = identity[0]
            self._ack_seconds[phase] = (
                self._ack_seconds.get(phase, 0.0) + perf_counter() - started
            )

    def _fail_ack(self, failure: Exception) -> NoReturn:
        self._failed = self._poisoned = True
        cause = None
        if isinstance(failure, ValueError):
            # All ranks completed the first all-gather, even if they entered
            # different phases (notably abort vs source_done). Rendezvous once
            # more with a constant identity, AFTER every local owner use drains.
            error = self._drain_devices()
            try:
                self._drain_required_remote()
            except Exception as exc:
                error = error or exc
            try:
                self._engine.layerwise_prefill_ack(
                    ("failed_step_drained", self._step), error
                )
            except ValueError as exc:
                # Storage/native submission errors can be reported after a safe
                # drain. The completed second all-gather still proves fencing.
                cause = exc
                self._abort_drained = True
            except Exception as exc:
                cause, failure = failure, exc
                self._unsafe_transfer = True
            else:
                self._abort_drained = True
        else:
            # Broken collectives and unknown device fences cannot safely enter
            # a new collective or release any rank's shared slab owners.
            self._unsafe_transfer = True
        if self._unsafe_transfer and not isinstance(
            failure, LayerwisePrefillFenceError
        ):
            cause = failure
            failure = LayerwisePrefillFenceError(
                f"Async failure drain is unknown: {failure}"
            )
        self._failure_error = failure
        if self._step_future is not None and not self._step_future.done():
            self._step_future.set_exception(failure)
        raise failure from cause

    def _storage_error(self, error: BaseException) -> Exception:
        """Normalize storage interruptions for TP propagation; worker must restart."""
        self._poisoned = True
        if isinstance(error, Exception):
            return error
        normalized = RuntimeError(
            f"Async storage {type(error).__name__}: {error}; restart worker"
        )
        normalized.__cause__ = error
        return normalized

    def _page_batches(
        self, req: LayerwisePrefillRequest, group: int, keep: int
    ) -> Iterator[range]:
        plan = self._plans[req.request_id, group, req.compute_end]
        base, size = keep, 0
        for chunk in range(keep, len(plan)):
            page_bytes = sum(
                self._prefixes[req.request_id, req.allocation_generation, group, row]
                .objects[chunk]
                .get_size()
                for row in range(self.layer_count(group))
            )
            if page_bytes > self._limits[1]:
                raise ValueError("Actual indivisible page exceeds max_bytes")
            if chunk > base and (
                chunk - base == 16 or size + page_bytes > self._limits[1]
            ):
                yield range(base, chunk)
                base, size = chunk, 0
            size += page_bytes
        if base < len(plan):
            yield range(base, len(plan))

    def _submit_required_remote(self, keys: list, objects: list[MemoryObj]) -> None:
        if self._queue is None:
            try:
                self._queue = RequiredPutQueue(
                    self._engine.storage_manager,
                    max_jobs=self._limits[0],
                    max_bytes=self._limits[1],
                    max_futures=self._limits[2],
                )
            except BaseException as exc:
                for obj in objects:
                    obj.ref_count_down()
                raise self._storage_error(exc) from exc
        try:
            self._queue.submit(keys, objects)
        except BaseException as exc:
            # RequiredPutQueue consumes loans and drains before any exit,
            # including KeyboardInterrupt/SystemExit from workers/admission.
            raise self._storage_error(exc) from exc
        self._remote_jobs += 1

    def _drain_required_remote(self) -> None:
        if self._queue is not None:
            try:
                self._queue.close()
            except BaseException as exc:
                raise self._storage_error(exc) from exc
