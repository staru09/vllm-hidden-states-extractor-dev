# SPDX-License-Identifier: Apache-2.0
"""
Hidden Activations Connector for vLLM.

Captures hidden activations from a configurable layer during inference,
stores them in a GPU buffer (no CPU sync), and returns a buffer handle
in the API response.

Architecture:
    1. Plugin patches the model class to register a forward hook on target layer
    2. Hook captures the layer output and stores in GPUBufferManager (stays on GPU)
    3. Connector's request_finished() returns the buffer handle in kv_transfer_params
    4. Consumer process reads the tensor using the handle (via CUDA IPC or co-process API)
"""

import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_hidden_states_extractor.gpu_buffer import (
    get_global_buffer,
    init_global_buffer,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# ─── Global state for hook <-> connector communication ───
# The hook writes here, the connector reads from here.
# No locks needed: single-threaded GPU execution in vLLM worker.
_pending_hidden_states: Dict[str, List[str]] = defaultdict(list)  # req_id -> [handle, ...]
_current_request_token_counts: Dict[str, int] = {}  # req_id -> num_tokens_in_batch
_layer_index: int = 20                         # configurable
_capture_enabled: bool = False


def _make_activation_hook(layer_idx: int):
    """
    Create a forward hook for a specific layer.

    The hook captures the layer output tensor, stores it in the
    GPU buffer manager, and records the handle for the current request.

    For each forward pass, the batch tensor is sliced per-request using
    _current_request_token_counts (populated from SchedulerOutput.
    num_scheduled_tokens). Each request gets its own tensor slice stored
    as a separate buffer handle.

    IMPORTANT: No locks, no CPU sync, no torch-unsupported ops.
    The tensor stays on GPU.

    NOTE: We call get_global_buffer() at runtime (not via closure)
    to ensure we always use the current buffer instance, even if
    init_global_buffer() was called after hook registration.
    """
    def hook(module, input, output):
        if not _capture_enabled:
            return
        if not _current_request_token_counts:
            return

        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Get the current buffer (not closure-captured)
        buffer = get_global_buffer()

        # Slice per-request using token counts.
        # _current_request_token_counts is ordered (Python 3.7+ dict),
        # matching the order tokens appear in the batch tensor.
        offset = 0
        for req_id, num_tokens in _current_request_token_counts.items():
            # Slice this request's tokens from the batch
            req_hidden = hidden_states[offset:offset + num_tokens]
            offset += num_tokens

            handle = buffer.store(
                req_hidden,
                metadata={
                    "req_id": req_id,
                    "layer_idx": layer_idx,
                    "shape": list(req_hidden.shape),
                    "dtype": str(req_hidden.dtype),
                    "is_prefill": num_tokens > 1,
                },
            )
            _pending_hidden_states[req_id].append(handle)

    return hook


def register_activation_hooks(model: torch.nn.Module, layer_idx: int):
    """
    Register a forward hook on the specified layer of the model.
    
    Args:
        model: The model to register hooks on
        layer_idx: Which layer to extract activations from (e.g., 20)
        
    Returns:
        List of hook handles
    """
    handles = []

    # Find the layers module
    layers = None
    for name, module in model.named_modules():
        if name == 'model.layers' or name.endswith('.model.layers'):
            layers = module
            logger.info(f"Found layers at: {name}")
            break

    if layers is None:
        for name, module in model.named_modules():
            if 'layers' in name and isinstance(module, torch.nn.ModuleList):
                layers = module
                logger.info(f"Found layers at: {name}")
                break

    if layers is None:
        logger.error("Could not find model layers for hook registration")
        return handles

    if layer_idx < len(layers):
        layer = layers[layer_idx]
        hook_handle = layer.register_forward_hook(_make_activation_hook(layer_idx))
        handles.append(hook_handle)
        logger.info(f"Hidden activations hook registered on layer {layer_idx}")
    else:
        logger.error(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")

    return handles


# ─── Connector ───

@dataclass
class ActivationsConnectorMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class HiddenActivationsConnector(KVConnectorBase_V1):
    """
    KV Connector that captures hidden activations from a model layer.
    
    Hidden states are stored in a GPU buffer (no CPU transfer).
    The buffer handle is returned in the API response via kv_transfer_params.
    
    Config (via kv_connector_extra_config):
        - activation_layer: int = 20   (which layer to capture)
        - buffer_size: int = 64        (max number of stored tensors)
        - buffer_ttl: float = 30.0     (seconds before auto-cleanup)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        global _layer_index, _capture_enabled

        self._block_size = vllm_config.cache_config.block_size

        # Read config
        _layer_index = self._kv_transfer_config.get_from_extra_config(
            "activation_layer", 20
        )
        buffer_size = self._kv_transfer_config.get_from_extra_config(
            "buffer_size", 64
        )
        buffer_ttl = self._kv_transfer_config.get_from_extra_config(
            "buffer_ttl", 30.0
        )

        # Initialize global buffer
        init_global_buffer(
            max_slots=buffer_size,
            default_ttl=buffer_ttl,
        )

        # Enable hooks via env var for model patching
        os.environ["HIDDEN_ACTIVATIONS_ENABLED"] = "1"
        os.environ["HIDDEN_ACTIVATIONS_LAYER"] = str(_layer_index)

        _capture_enabled = True

        logger.info(
            f"HiddenActivationsConnector initialized: "
            f"layer={_layer_index}, buffer_size={buffer_size}, ttl={buffer_ttl}s"
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches."""
        logger.info(f"Registered {len(kv_caches)} KV cache layers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Track ALL active requests so the hook captures every step.

        Uses num_scheduled_tokens which includes both new prefill requests
        and ongoing decode requests, with their token counts per step.
        """
        global _current_request_token_counts
        meta = ActivationsConnectorMetadata()

        # num_scheduled_tokens: dict[str, int] — covers all active requests
        _current_request_token_counts = dict(scheduler_output.num_scheduled_tokens)

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return all buffer handles for the captured hidden states.

        Returns a list of handles — one per forward step (prefill + each
        decode step). The consumer can iterate through them to get the
        full sequence of hidden states.
        """
        global _current_request_token_counts

        req_id = request.request_id
        handles = list(_pending_hidden_states.pop(req_id, []))

        # Remove from active tracking
        _current_request_token_counts.pop(req_id, None)

        if handles:
            # Peek at the first handle for dtype/layer info
            buffer = get_global_buffer()
            _, first_meta = buffer.get(handles[0])
            dtype = first_meta.get("dtype", "") if first_meta else ""

            logger.info(
                f"Request {req_id}: {len(handles)} hidden state steps captured, "
                f"dtype={dtype}, layer={_layer_index}"
            )

            return False, {
                "hidden_states_handles": handles,
                "hidden_states_num_steps": len(handles),
                "hidden_states_dtype": dtype,
                "hidden_states_layer": _layer_index,
                "hidden_states_device": "gpu",
            }
        else:
            logger.warning(f"No hidden states captured for request {req_id}")
            return False, {"hidden_states_handles": []}

    def clear_connector_metadata(self):
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
