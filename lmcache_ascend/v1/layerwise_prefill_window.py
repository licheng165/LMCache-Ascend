# SPDX-License-Identifier: Apache-2.0
"""Stage 4 layerwise-prefill NPU transfer-window backend.

The generic LMCache connector owns the protocol state machine (per-group
cursors, allocation generations, the SAVE_SUBMITTED -> SOURCE_DONE ->
PERSIST_DONE lifecycle and the bounded pending-save queue). This module
provides the device-side counterpart consumed through
``engine.layerwise_prefill_window_backend``: per-group, per-bank D2H/H2D
event ledgers, group-local row/bank validation, and the pre-HCOM submit /
post-HCOM publish split around the projection and all-reduce window.

The actual NPU kernels are delegated to a :class:`LayerwisePrefillDeviceOps`
implementation (bound to the layerwise NPU connector). The engine only
advertises the backend when the connector opts in with
``supports_layerwise_prefill_window() is True``, so a connector without the
ops wiring keeps every transfer-window capability False and the callbacks
fail closed before any device launch.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache_ascend.v1.dsa_kv_topology import DSAKVTopologyView

LATENT_PLANE_COUNT = 2
INDEXER_PLANE_COUNT = 1


class LayerwisePrefillDeviceOps:
    """Device-side submission hooks for one transfer window.

    All waits are stream-side: ``wait_event`` must be waited on the
    submission stream before the kernel enqueues, and ``done_event`` is
    recorded on that stream after the kernel. No method may block the host
    during the pre-HCOM phase.
    """

    def layout_signature(self) -> Optional[str]:
        """Return the live connector layout signature.

        The backend re-checks it before every launch; a rebind after
        construction invalidates the window and must fail closed.
        """
        return None

    def source_bank_count(self) -> int:
        """Number of physical staging source banks (1 or 2)."""
        return 1

    def submit_save(
        self,
        group: int,
        row_ordinal: int,
        bank: int,
        kv_planes: list[torch.Tensor],
        attn_metadata: Any,
        wait_event: Optional[Any] = None,
        done_event: Optional[Any] = None,
    ) -> None:
        """Pre-HCOM: enqueue the D2H for one canonical row (no host work)."""
        raise NotImplementedError

    def submit_load(
        self,
        group: int,
        row_ordinal: int,
        bank: int,
        metadata: Any,
        wait_event: Optional[Any] = None,
        done_event: Optional[Any] = None,
    ) -> None:
        """Pre-HCOM: enqueue the H2D for the next row of one group."""
        raise NotImplementedError

    def wait_for_load(
        self,
        group: int,
        row_ordinal: int,
        bank: int,
        metadata: Any,
    ) -> None:
        """Make the row's required prefix ready on the compute stream.

        With no prior load submission, restore the prefix synchronously,
        including bootstrap and the next prefill chunk's own prefix. Raise
        if that prefix cannot be restored; absence of an event is not proof
        that the prefix is empty.
        """
        raise NotImplementedError

    def finish_publish(
        self,
        metadata: Any,
        done_event: Any,
    ) -> Optional[Any]:
        """Post-HCOM: publish one submitted row save.

        ``done_event`` was recorded on the D2H stream; host-side publication
        may now proceed. Returns a completion-required backend future or
        None when the store committed synchronously.
        """
        raise NotImplementedError

    def sync_save(
        self,
        group: int,
        row_ordinal: int,
        bank: int,
        kv_planes: list[torch.Tensor],
        attn_metadata: Any,
    ) -> None:
        """Stage 3 contract: fully save one row in one call."""
        raise NotImplementedError

    def abort_request(self, request_id: str) -> None:
        """Clean device resources owned by one aborted request."""
        raise NotImplementedError


@dataclass
class _BankLedger:
    save_done: Any = None
    load_done: Any = None
    load_identity: Optional[tuple[int, tuple[tuple[str, int], ...]]] = None
    ready_identity: Optional[tuple[int, tuple[tuple[str, int], ...]]] = None


class LayerwisePrefillNPUWindowBackend:
    """Per-group bank events and launch validation for the P-node window.

    The backend serves both the synchronous Stage 3 contract and the Stage 4
    pre/post-HCOM contract:

    - Each group owns independent save/load completion events per physical
      bank, so LATENT and INDEXER transfers never share synchronization
      state.
    - A load into a bank waits for that bank's previous save D2H before
      overwriting it; the compute stream waits for the load event before
      consuming a row (``wait_for_load``).
    - Direct paths may use two source banks per group; single-staging
      paths serialize saves into their one staging tensor instead of
      pretending to have two banks.
    - Generation identity is validated before launch; stale-generation
      completions never publish or touch current banks. Release retires
      request-owned records without discarding physical bank dependencies.
    """

    def __init__(
        self,
        view: DSAKVTopologyView,
        ops: LayerwisePrefillDeviceOps,
        *,
        event_factory: Any = None,
    ):
        self._view = view
        self._ops = ops
        self._event_factory = event_factory or (
            lambda: torch.npu.Event()  # type: ignore[attr-defined]
        )
        self._construction_layout = ops.layout_signature()
        self._ledgers: dict[int, dict[int, _BankLedger]] = {}
        # group -> row -> (generations, D2H done event)
        self._submitted: dict[
            int, dict[int, tuple[tuple[tuple[str, int], ...], Any]]
        ] = {}
        # group -> most recent submitted D2H event; single-staging paths
        # wait on it before reusing their one staging tensor.
        self._group_last_submit_event: dict[int, Any] = {}
        self._active_generations: dict[str, int] = {}
        # One high-water mark per released request, not a permanent ID ban.
        self._released_generations: dict[str, int] = {}

    @property
    def supports_sync_callbacks(self) -> bool:
        return self._has_concrete_hooks(("sync_save", "wait_for_load", "abort_request"))

    @property
    def supports_transfer_window(self) -> bool:
        return self._has_concrete_hooks(
            (
                "submit_save",
                "submit_load",
                "wait_for_load",
                "finish_publish",
                "abort_request",
            )
        )

    @property
    def persists_indexer_group(self) -> bool:
        return bool(self._view.layer_counts[1])

    @property
    def topology_signature(self) -> str:
        return self._view.signature

    def layer_count(self, group: int) -> int:
        if group not in (0, 1):
            raise ValueError(f"Invalid DSA KV group {group}.")
        return self._view.layer_counts[group]

    def _row(
        self,
        metadata: Any,
    ) -> tuple[tuple[str, int, int, int, int], int, int, int]:
        row = metadata.row
        try:
            key = (
                row.layer_name,
                row.execution_ordinal,
                row.kv_group,
                row.row_ordinal,
                row.bank,
            )
        except AttributeError as exc:
            raise ValueError(
                "Layerwise-prefill callback metadata is missing row fields."
            ) from exc
        group = key[2]
        row_ordinal = key[3]
        if group not in (0, 1):
            raise ValueError(f"Invalid DSA KV group {group}.")
        rows = self._view.rows_by_group[group]
        if row_ordinal >= len(rows) or rows[row_ordinal] != key:
            raise ValueError(
                "Layerwise-prefill row is absent from the frozen NPU layout: "
                f"{key!r}."
            )
        if key[4] != row_ordinal % 2:
            raise ValueError(
                "Layerwise-prefill row bank disagrees with group-local row "
                f"parity: {key!r}."
            )
        return key, group, row_ordinal, key[4]

    def _request_generations(self, metadata: Any) -> tuple[tuple[str, int], ...]:
        self._check_layout_rebind()
        generations = []
        request_ids = set()
        for request_id, generation in metadata.request_generations:
            if (
                not isinstance(request_id, str)
                or not request_id
                or not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation <= 0
                or request_id in request_ids
            ):
                raise ValueError(
                    "Layerwise-prefill backend requires valid request "
                    "generations before any device launch."
                )
            request_ids.add(request_id)
            generations.append((request_id, generation))
        if not generations:
            raise ValueError("Layerwise-prefill backend requires request generations.")
        return tuple(generations)

    def _validate_launch(self, metadata: Any) -> tuple[tuple[str, int], ...]:
        generations = self._request_generations(metadata)
        for request_id, generation in generations:
            if (
                generation < self._active_generations.get(request_id, 0)
                or generation <= self._released_generations.get(request_id, 0)
            ):
                raise ValueError(
                    "Layerwise-prefill backend refuses a superseded or released "
                    f"request generation: {request_id!r}/{generation}."
                )
        # Activation follows successful device submission/readiness. A failed
        # newer restore must not invalidate the coordinator's older saves.
        return generations

    def _check_layout_rebind(self) -> None:
        live = self._ops.layout_signature()
        if live != self._construction_layout:
            raise ValueError(
                "Layerwise-prefill connector layout was rebound after "
                f"window construction: {self._construction_layout!r} -> "
                f"{live!r}."
            )

    def _ledger(self, group: int, bank: int) -> _BankLedger:
        ledgers = self._ledgers.setdefault(group, {})
        ledger = ledgers.get(bank)
        if ledger is None:
            ledger = _BankLedger()
            ledgers[bank] = ledger
        return ledger

    def _kv_planes(self, group: int, kv_layer: Any) -> list[torch.Tensor]:
        tensors = (
            list(kv_layer)
            if isinstance(kv_layer, (list, tuple))
            else [kv_layer]
        )
        expected = LATENT_PLANE_COUNT if group == 0 else INDEXER_PLANE_COUNT
        if len(tensors) != expected or any(
            not torch.is_tensor(tensor) for tensor in tensors
        ):
            raise ValueError(
                "Layerwise-prefill save has the wrong KV plane count for "
                f"group={group}: expected {expected}, got {len(tensors)}."
            )
        return tensors

    def _source_bank_count(self) -> int:
        banks = int(self._ops.source_bank_count())
        if banks not in (1, 2):
            raise ValueError(
                "Layerwise-prefill connector declared an invalid source bank "
                f"count {banks}; expected 1 or 2."
            )
        return banks

    def wait_for_load(self, metadata: Any) -> None:
        """Restore/wait for a row, including sync and first-row bootstrap loads.

        Stale generations raise before device work. Readiness is recorded only
        after the device hook succeeds, and is invalidated by the next save/load.
        """

        key, group, row_ordinal, bank = self._row(metadata)
        generations = self._validate_launch(metadata)
        ledger = self._ledger(group, bank)
        identity = (row_ordinal, generations)
        if ledger.load_identity is not None and ledger.load_identity != identity:
            raise ValueError("Layerwise-prefill wait does not own the pending load.")
        if ledger.ready_identity == identity:
            return
        # No ledger is not proof of an empty prefix: the ops must restore it
        # synchronously or fail closed, never silently skip bootstrap.
        self._ops.wait_for_load(group, row_ordinal, bank, metadata)
        self._active_generations.update(generations)
        ledger.load_done = None
        ledger.load_identity = None
        ledger.ready_identity = identity

    def submit_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
    ) -> None:
        """Pre-HCOM: validate and enqueue one canonical row D2H save."""

        key, group, row_ordinal, bank = self._row(metadata)
        planes = self._kv_planes(group, kv_layer)
        if row_ordinal in self._submitted.get(group, {}):
            raise ValueError(
                "Layerwise-prefill row was already submitted before its "
                f"finish: group={group}, row={row_ordinal}."
            )
        if self._source_bank_count() == 1:
            # Single staging: the previous D2H into the shared staging
            # tensor must have drained before it can be reused.
            wait_event = self._group_last_submit_event.get(group)
        else:
            wait_event = None
        generations = self._validate_launch(metadata)
        done_event = self._event_factory()
        self._ops.submit_save(
            group,
            row_ordinal,
            bank,
            planes,
            attn_metadata,
            wait_event=wait_event,
            done_event=done_event,
        )
        self._active_generations.update(generations)
        self._group_last_submit_event[group] = done_event
        self._submitted.setdefault(group, {})[row_ordinal] = (generations, done_event)
        # Physical bank dependencies survive supersession and release; a late
        # finish must not replace the event of a newer save into this bank.
        ledger = self._ledger(group, bank)
        ledger.save_done = done_event
        ledger.ready_identity = None

    def submit_load(self, metadata: Any) -> None:
        """Pre-HCOM: enqueue next-row H2D for every present group."""

        generations = self._validate_launch(metadata)
        execution = metadata.execution
        groups = (0,) if execution.indexer is None else (0, 1)
        loads = []
        for group in groups:
            row = execution.latent if group == 0 else execution.indexer
            assert row is not None
            next_row = int(row.row_ordinal) + 1
            if next_row >= self.layer_count(group):
                continue
            bank = next_row % 2
            ledger = self._ledger(group, bank)
            identity = (next_row, generations)
            if ledger.load_identity is not None:
                if ledger.load_identity != identity:
                    raise ValueError(
                        "Layerwise-prefill load would replace a pending load."
                    )
                continue
            loads.append((group, next_row, bank, ledger, identity))
        for group, next_row, bank, ledger, identity in loads:
            # Load stream must wait for the previous save into this bank
            # before overwriting it.
            done_event = self._event_factory()
            self._ops.submit_load(
                group,
                next_row,
                bank,
                metadata,
                wait_event=ledger.save_done,
                done_event=done_event,
            )
            ledger.load_done = done_event
            ledger.load_identity = identity
            ledger.ready_identity = None
        self._active_generations.update(generations)

    def finish_save(self, metadata: Any) -> Optional[Any]:
        """Post-HCOM: publish one submitted save and return its persist future."""

        key, group, row_ordinal, bank = self._row(metadata)
        generations = self._request_generations(metadata)
        record = self._submitted.get(group, {}).get(row_ordinal)
        if any(
            generation < self._active_generations.get(request_id, 0)
            or generation <= self._released_generations.get(request_id, 0)
            for request_id, generation in generations
        ):
            if record is not None and record[0] == generations:
                del self._submitted[group][row_ordinal]
            return None
        if record is None or any(
            generation != self._active_generations.get(request_id)
            for request_id, generation in generations
        ):
            raise ValueError(
                "Layerwise-prefill finish arrived before its submit: "
                f"group={group}, row={row_ordinal}."
            )
        submitted_generations, done_event = record
        if submitted_generations != generations:
            raise ValueError(
                "Layerwise-prefill finish identity differs from its submit."
            )
        future = self._ops.finish_publish(metadata, done_event)
        del self._submitted[group][row_ordinal]
        return future

    def sync_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
    ) -> None:
        """Stage 3 contract: fully save one row in one call."""

        key, group, row_ordinal, bank = self._row(metadata)
        planes = self._kv_planes(group, kv_layer)
        generations = self._validate_launch(metadata)
        self._ops.sync_save(group, row_ordinal, bank, planes, attn_metadata)
        self._active_generations.update(generations)
        self._ledger(group, bank).ready_identity = None

    def abort_request(self, request_id: str) -> None:
        """Release request-owned records, blocking only known old generations."""

        generation = self._active_generations.pop(request_id, 0)
        self._released_generations[request_id] = max(
            generation, self._released_generations.get(request_id, 0)
        )
        self._ops.abort_request(request_id)
        for rows in self._submitted.values():
            for row, (generations, _) in tuple(rows.items()):
                if any(req_id == request_id for req_id, _ in generations):
                    del rows[row]
        for ledgers in self._ledgers.values():
            for ledger in ledgers.values():
                # Retain physical events after request identity is retired.
                if ledger.load_identity is not None and any(
                    req_id == request_id for req_id, _ in ledger.load_identity[1]
                ):
                    ledger.load_identity = None
                if ledger.ready_identity is not None and any(
                    req_id == request_id for req_id, _ in ledger.ready_identity[1]
                ):
                    ledger.ready_identity = None

    def _has_concrete_hooks(self, names: tuple[str, ...]) -> bool:
        for name in names:
            hook = getattr(self._ops, name, None)
            if (
                not callable(hook)
                or getattr(hook, "__func__", hook)
                is getattr(LayerwisePrefillDeviceOps, name)
                or getattr(hook, "__isabstractmethod__", False)
            ):
                return False
        return True
