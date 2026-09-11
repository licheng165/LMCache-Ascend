# SPDX-License-Identifier: Apache-2.0
"""Single-emitter protocol/publication thread for the async transfer window.

Plan B: while a step is bound, this thread is the only emitter of this rank's
lmcache CPU-group collectives — shared-handle envelope broadcasts (row
publication and bootstrap resolution), the device_finish/finish/commit gates
and the abort/failure handshakes. The model thread keeps validation, planning,
slot preparation, D2H/H2D enqueue and device fences, and is always blocked on
a descriptor Future, a capacity wait or its own device work whenever this
thread touches shared step state, so the two threads never race the backend.

The queue is strictly FIFO and the model thread enqueues the identical
descriptor sequence on every TP rank, so every rank's protocol thread enters
the CPU group's collectives in the same order. Descriptor payloads are
transport data only; all step logic stays on the backend whose private
``_protocol_*`` methods this thread dispatches to.

Failure protocol: a failed publication returns its failure (the root cause
already rode the shared-handle error envelopes, so peers observe the same
failure at the same sequence position); any other descriptor failure raises
through dispatch exactly as the model thread would have raised inline. In
both cases the thread records the failure, cancels every not-yet-started
descriptor and keeps serving only abort/stop descriptors, so a later
abort_devices rendezvous still meets every rank's thread at the same
collective position.
"""

# Standard
from concurrent.futures import Future, wait as wait_futures
from threading import Condition, Thread
from typing import Any
import queue
import weakref

# First Party
from lmcache_ascend.v1.layerwise_prefill_sync import (
    LayerwisePrefillSyncBackend,
)

_STOP = object()


class _Descriptor:
    """One frozen work item; the payload is transport data only."""

    __slots__ = ("kind", "payload", "future")

    def __init__(self, kind: str, payload: Any = None) -> None:
        self.kind = kind
        self.payload = payload
        self.future = Future()


class LayerwisePrefillProtocolThread:
    """Own this rank's lmcache CPU-group collectives for one async backend."""

    def __init__(self, backend: Any) -> None:
        # Weak reference: a discarded backend must stay collectable even with
        # its emitter alive. Production backends live for the worker process.
        self._backend = weakref.ref(backend)
        context = getattr(backend._engine, "protocol_thread_context", None)
        self._context = context if callable(context) else None
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._condition = Condition()
        self._queued = 0
        self._publish_jobs = 0
        self._publish_bytes = 0
        self._failure: BaseException | None = None
        self._cancelled = False
        self._thread = Thread(
            target=self._run,
            name="layerwise-prefill-protocol",
            daemon=True,
        )
        self._thread.start()

    # ---- model-thread API -------------------------------------------------

    def _owner(self) -> Any:
        backend = self._backend()
        if backend is None:
            raise RuntimeError("Layerwise-prefill protocol backend was discarded")
        return backend

    def submit(self, kind: str, payload: Any = None) -> Future:
        """Enqueue one descriptor; publication admission is already paid for."""
        descriptor = _Descriptor(kind, payload)
        with self._condition:
            self._queued += 1
            if kind == "publish":
                record = payload[0]
                self._publish_jobs += 1
                self._publish_bytes += record.bytes
        self._queue.put(descriptor)
        return descriptor.future

    @staticmethod
    def wait(future: Future) -> Any:
        """Block until the descriptor completes; re-raise its failure."""
        # Wait through concurrent.futures so instrumentation that patches
        # Future.result to assert completion never sees a pending future.
        wait_futures((future,))
        return future.result()

    def await_publication_capacity(self, byte_cost: int) -> None:
        """Block while the admitted publication window would exceed limits.

        The window limits keep their existing meaning, but admission moves
        from row preparation to publication enqueue: at most ``max_jobs``
        unpublished rows and ``max_bytes`` unpublished payload bytes may sit
        in this queue. Returns immediately once a failure cancels the queue;
        the descriptor submitted afterwards is failed like the rest.
        """
        jobs_limit, bytes_limit, _ = self._owner()._limits
        with self._condition:
            while not self._cancelled and (
                self._publish_jobs + 1 > jobs_limit
                or self._publish_bytes + byte_cost > bytes_limit
            ):
                self._condition.wait()

    @property
    def queued(self) -> int:
        with self._condition:
            return self._queued

    def join_publications(self) -> None:
        """Block until every admitted publication descriptor completed."""
        with self._condition:
            while self._publish_jobs:
                self._condition.wait()

    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "publish_jobs": self._publish_jobs,
                "publish_bytes": self._publish_bytes,
                "queued": self._queued,
                "cancelled": self._cancelled,
            }

    def close(self, timeout: float = 10.0) -> None:
        """Stop serving descriptors; never drain storage or device owners."""
        self._queue.put(_STOP)
        self._thread.join(timeout)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                self._fail_future(
                    item, RuntimeError("Layerwise-prefill protocol thread closed")
                )

    # ---- protocol-thread side ---------------------------------------------

    def _run(self) -> None:
        if self._context is not None:
            try:
                self._context()
            except BaseException:
                pass  # Test-only thread context must never kill the emitter.
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._drain_stopped()
                return
            with self._condition:
                # Count from dispatch start, not completion: step-end logging
                # inside the finish descriptor must observe an empty queue,
                # and the bind-time drain check joins on the Future anyway.
                self._queued -= 1
            try:
                if self._cancelled and item.kind != "abort":
                    self._cancel_descriptor(item)
                    continue
                result = self._dispatch(item)
            except BaseException as exc:
                self._fail(item, exc)
            else:
                self._finish(item)
                if not item.future.done():
                    item.future.set_result(result)

    def _dispatch(self, item: _Descriptor) -> Any:
        backend = self._owner()
        kind = item.kind
        if kind == "publish":
            record, failure = item.payload
            returned = backend._protocol_publish(record, failure)
            if returned is not None:
                self._record_failure(returned)
                backend._pending_error = (
                    backend._pending_error or backend._callback_failure(returned)
                )
            return None
        if kind == "resolve":
            backend._protocol_resolve_row(*item.payload)
            return None
        if kind == "finish_step":
            backend._ack(("device_finish",), item.payload)
            LayerwisePrefillSyncBackend.finish_step(backend)
            return None
        if kind == "abort":
            with self._condition:
                # Skip stragglers enqueued behind this abort; peers truncate
                # their queues at the same failure position.
                self._cancelled = True
            backend._ack(("abort_devices",), item.payload)
            return None
        raise ValueError(f"Unknown layerwise-prefill protocol kind: {kind!r}")

    def _finish(self, item: _Descriptor) -> None:
        with self._condition:
            if item.kind == "publish":
                self._publish_jobs -= 1
                self._publish_bytes -= item.payload[0].bytes
            self._condition.notify_all()

    def _fail(self, item: _Descriptor, exc: BaseException) -> None:
        self._record_failure(exc)
        normalized = self._normalize(exc)
        backend = self._backend()
        if backend is not None:
            backend._pending_error = backend._pending_error or normalized
        self._finish(item)
        self._fail_future(item, normalized)

    def _cancel_descriptor(self, item: _Descriptor) -> None:
        with self._condition:
            failure = self._failure
        normalized = self._normalize(
            failure
            if failure is not None
            else RuntimeError("Layerwise-prefill protocol descriptor cancelled")
        )
        self._finish(item)
        self._fail_future(item, normalized)

    def _normalize(self, exc: BaseException) -> Exception:
        """Give thread failures the model thread's inline raise shape."""
        backend = self._backend()
        if backend is not None:
            return backend._callback_failure(exc)
        if isinstance(exc, Exception):
            return exc
        return RuntimeError(f"{type(exc).__name__}: {exc}")

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = exc
            self._cancelled = True

    @staticmethod
    def _fail_future(item: _Descriptor, exc: Exception) -> None:
        if not item.future.done():
            item.future.set_exception(exc)

    def _drain_stopped(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                self._fail_future(
                    item, RuntimeError("Layerwise-prefill protocol thread closed")
                )
