# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based hidden states connector for vLLM.

Works with the model_wrapper hooks to save captured hidden states.
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, List

import torch
from safetensors.torch import save_file

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_hidden_states_extractor.model_wrapper import (
    get_captured_states,
    clear_captured_states,
    set_current_request,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class HookBasedConnectorMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class HookBasedHiddenStatesConnector(KVConnectorBase_V1):
    """
    A KV connector that works with forward hooks to capture hidden states.
    
    The hooks are registered by the patched Qwen3 model __init__.
    This connector handles:
    - Setting up request info before inference
    - Saving captured states after request completion
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
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp/hidden_states"
        )
        self._layer_indices = self._kv_transfer_config.get_from_extra_config(
            "layer_indices", [7, 14, 21, 28]
        )
        
        self._request_filenames: dict[str, str] = {}
        self._request_token_ids: dict[str, list[int]] = {}
        
        # Set environment variable to enable hooks in model
        os.environ["ENABLE_HIDDEN_STATES_HOOKS"] = "1"
        os.environ["HIDDEN_STATES_LAYER_INDICES"] = ",".join(str(x) for x in self._layer_indices)
        
        logger.info(f"HookBasedHiddenStatesConnector initialized: path={self._storage_path}, layers={self._layer_indices}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches."""
        logger.info(f"Registered KV caches: {len(kv_caches)} layers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading KV cache (not used)."""
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load (not used)."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Save KV cache (not used - hooks capture directly)."""
        pass

    def wait_for_save(self):
        """Wait for save to complete."""
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get number of tokens that can be loaded (store-only, always 0)."""
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Update state after block allocation."""
        pass

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Build connector metadata for this step."""
        meta = HookBasedConnectorMetadata()
        
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            req_id = new_req.req_id
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")
            
            self._request_filenames[req_id] = filename
            self._request_token_ids[req_id] = token_ids
            
            # Set current request info for hooks to use
            set_current_request(req_id, token_ids, self._storage_path)
            
            # Clear any previously captured states
            clear_captured_states()
        
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes - save captured hidden states."""
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        token_ids = self._request_token_ids.pop(req_id, None)
        
        # Get captured hidden states
        captured = get_captured_states()
        
        if captured and req_filename:
            os.makedirs(self._storage_path, exist_ok=True)
            
            tensors = {}
            for layer_idx, states_list in sorted(captured.items()):
                if states_list:
                    stacked = torch.cat(states_list, dim=0)
                    tensors[f"layer_{layer_idx}"] = stacked
            
            if token_ids:
                tensors["token_ids"] = torch.tensor(token_ids, dtype=torch.long)
            
            if tensors:
                save_file(tensors, req_filename)
                logger.info(f"Saved hidden states from {len(captured)} layers to {req_filename}")
            
            clear_captured_states()
        else:
            logger.warning(f"No hidden states captured for request {req_id}")
        
        return False, {"hidden_states_path": req_filename}

    def clear_connector_metadata(self):
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
