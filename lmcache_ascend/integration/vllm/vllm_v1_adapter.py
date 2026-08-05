# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
)
from lmcache.logging import init_logger
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        self.store_async = self.config.store_async
        if (
            role != KVConnectorRole.SCHEDULER
            and self.kv_role != "kv_consumer"
            and self.use_layerwise
            and self.store_async
        ):
            raise ValueError(
                "Layerwise storing is not supported with async store"
            )
        logger.debug("store_async: %s", self.store_async)

    def _effective_skip_leading_tokens(
        self,
        _request: ReqMeta,
        save_spec: Any,
    ) -> int:
        # Ascend chunked prefill must not use producer transfer progress here.
        return save_spec.skip_leading_tokens

    def _prepare_direct_store_inputs(
        self,
        _request: ReqMeta,
        slot_mapping: torch.Tensor,
        save_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self.lmcache_engine is not None
        ordering_event = save_context.get("ordering_event")
        if ordering_event is None:
            ordering_event = torch.npu.Event()
            ordering_event.record()
            save_context["ordering_event"] = ordering_event

        if slot_mapping.device.type == "npu":
            slot_mapping_npu = slot_mapping.to(dtype=torch.long)
        else:
            slot_mapping = slot_mapping.pin_memory()
            with torch.npu.stream(
                self.lmcache_engine.gpu_connector.store_stream
            ):
                slot_mapping_npu = slot_mapping.to(
                    device="npu",
                    dtype=torch.long,
                    non_blocking=True,
                )
        return slot_mapping, {
            "ordering_event": ordering_event,
            "slot_mapping_npu": slot_mapping_npu,
        }

    def _finish_save_batch(self, _save_context: dict[str, Any]) -> None:
        super()._finish_save_batch(_save_context)
        if self.kv_role != "kv_consumer" and self.lmcache_engine is not None:
            try:
                self.lmcache_engine.wait_for_pending_sync_stores()
            except Exception as error:
                if not getattr(self, "_dsa_store_progress", None):
                    raise
                raise RuntimeError("DSA backend future failed") from error
        if getattr(self, "_dsa_store_progress", None):
            self._fence_dsa_npu_streams()

    def _fence_dsa_npu_streams(self) -> None:
        """Fence the current NPU stream and connector-owned transfer streams."""
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "current_stream"):
            raise RuntimeError("NPU stream support is unavailable for DSA fencing")
        streams = [torch.npu.current_stream()]
        engine = self.lmcache_engine
        connector = (
            getattr(engine, "gpu_connector", None)
            if engine is not None
            else None
        )
        for name in ("store_stream", "load_stream", "broadcast_stream"):
            stream = getattr(connector, name, None)
            if stream is not None:
                streams.append(stream)
        synchronized: set[int] = set()
        for stream in streams:
            if id(stream) in synchronized:
                continue
            synchronize = getattr(stream, "synchronize", None)
            if not callable(synchronize):
                raise RuntimeError(f"DSA NPU stream {stream!r} cannot be fenced")
            synchronize()
            synchronized.add(id(stream))

    def _fence_dsa_source_activation(self) -> None:
        self._fence_dsa_npu_streams()

    def _fence_dsa_source_use(self) -> None:
        self._fence_dsa_npu_streams()

    def _handle_save_request_error(
        self,
        request: ReqMeta,
        _error: Exception,
    ) -> bool:
        logger.exception(
            "wait_for_save failed for request %s; skipping save",
            request.req_id,
        )
        return True

    def _finalize_worker_requests_after_store(
        self, finished_req_ids: set[str]
    ) -> set[str]:
        """Release worker state once its store pipeline no longer owns it."""
        if self.lmcache_engine is None:
            return super()._finalize_worker_requests_after_store(
                finished_req_ids
            )
        finished_sending = set(
            self.lmcache_engine.get_finished_stores(finished_req_ids) or ()
        )
        releasable_req_ids = (
            finished_sending if self.store_async else finished_req_ids
        )
        if releasable_req_ids:
            self._release_finished_worker_requests(releasable_req_ids)
        return finished_sending

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.lmcache_engine is None:
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )

        if self.store_async and self.kv_role != "kv_consumer":
            waited_req_ids = self.lmcache_engine.wait_for_pending_stores(
                preempted_req_ids
            )
            if waited_req_ids:
                logger.info(
                    "Handled preemptions after draining async stores: req_ids=%s",
                    sorted(waited_req_ids),
                )

        # Do not release generation-owned CPU objects until all NPU/connector
        # streams and source leases are proven quiescent. Legacy preemption
        # remains unchanged when no generation-bound source exists.
        has_dsa_sources = any(
            source.request_key.request_id in preempted_req_ids
            for source in getattr(self, "_dsa_sealed_sources", {}).values()
        )
        if has_dsa_sources:
            self._fence_dsa_npu_streams()
        self._release_dsa_source_leases_for_requests(preempted_req_ids)
        for req_id in preempted_req_ids:
            self._drop_dsa_worker_state(req_id)
        # NOTE: a full typed preemption quiesce additionally requires the NPU
        # runner to stop in-flight retrieve/store DMA and emit a
        # ``preemption_quiesce_ready`` receipt the Scheduler waits on before
        # freeing both groups' blocks. That runner-side fence is wired in the
        # vLLM-Ascend model runner; this connector side drops its worker state
        # and drains pending stores so no freed block is re-read.

    def _drop_dsa_worker_state(self, req_id: str) -> None:
        """Drop DSA prepared-source / lease state for a request.

        Idempotent; safe to call when no DSA state exists (threshold routing
        disabled).  Mirrors the retrieve-state drop so preemption of a SPARSE
        request cannot leave a dangling prepared source pointing at freed
        blocks.
        """
        if self._retire_dsa_request_sources(req_id):
            self._drop_worker_retrieve_state(req_id)

    def get_request_route_state(self, req_id: str) -> Optional[str]:
        """Return the authoritative DSA route state string for a request.

        Workers/runners query this instead of re-deriving sparse mode from
        prompt_len (design section 10.1).  Returns None when no snapshot was
        published for the request (LEGACY / threshold disabled).
        """
        trackers = getattr(self, "_request_trackers", None)
        if trackers is None:
            return None
        tracker = trackers.get(req_id)
        if tracker is None:
            return None
        return getattr(tracker, "dsa_route_state", None)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        _, return_params = super().request_finished(request, block_ids)
        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
