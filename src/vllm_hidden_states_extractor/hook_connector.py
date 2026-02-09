# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based hidden states connector for vLLM.

This connector integrates with vLLM's KV connector system to trigger
hidden states extraction at the right times during inference.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from safetensors.torch import save_file

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_hidden_states_extractor.hooks import get_hook_manager

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class HookBasedHiddenStatesConnector(KVConnectorBase_V1):
    """
    A KV connector that uses forward hooks to extract hidden states.
    
    This connector:
    1. Registers hooks on model layers when initialized
    2. Saves hidden states after each request completes
    3. Returns the save path in the response
    """
    
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
    ):
        super().__init__(vllm_config, role)
        
        # Get configuration from kv_connector_extra_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.save_path = extra_config.get("shared_storage_path", "/tmp/hidden_states")
        self.layer_indices = extra_config.get("layer_indices", [7, 14, 21, 28])
        
        # Configure the hook manager
        hook_manager = get_hook_manager()
        hook_manager.configure(self.layer_indices, self.save_path)
        
        self._model_initialized = False
        self._pending_requests = {}
        
        logger.info(f"HookBasedHiddenStatesConnector initialized: layers={self.layer_indices}")
    
    def register_hooks_on_model(self, model: torch.nn.Module):
        """Register hooks on the model. Should be called after model is loaded."""
        if self._model_initialized:
            return
            
        hook_manager = get_hook_manager()
        hook_manager.register_hooks(model)
        self._model_initialized = True
        logger.info("Hooks registered on model")
    
    def bind_connector_metadata(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "SchedulerOutput":
        """Called before forward pass - set up for this batch of requests."""
        # Track request IDs in this batch
        for req in scheduler_output.scheduled_new_reqs:
            hook_manager = get_hook_manager()
            hook_manager.set_request_id(req.req_id)
            self._pending_requests[req.req_id] = {
                "token_ids": req.prompt_token_ids if hasattr(req, 'prompt_token_ids') else None
            }
        
        return scheduler_output
    
    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[dict]:
        """Build metadata to be returned in the response."""
        # Nothing special needed here
        return None
    
    def save_for_request(self, request_id: str, token_ids: Optional[torch.Tensor] = None) -> Optional[str]:
        """Save hidden states for a specific request."""
        hook_manager = get_hook_manager()
        hook_manager.set_request_id(request_id)
        return hook_manager.save_hidden_states(token_ids)
    
    def real_clear_connector_metadata(self) -> None:
        """Clear connector metadata after saving."""
        # Save any pending hidden states
        hook_manager = get_hook_manager()
        for req_id, req_info in self._pending_requests.items():
            token_ids = req_info.get("token_ids")
            if token_ids is not None:
                token_ids = torch.tensor(token_ids, dtype=torch.long)
            save_path = hook_manager.save_hidden_states(token_ids)
            if save_path:
                logger.info(f"Saved hidden states for request {req_id}: {save_path}")
        
        self._pending_requests.clear()
    
    # Required interface methods
    def get_num_new_matched_requests(self) -> int:
        return 0
    
    def get_finished(
        self,
        finished_req_ids: List[str],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        # Save hidden states for finished requests
        saved_paths = []
        for req_id in finished_req_ids:
            if req_id in self._pending_requests:
                req_info = self._pending_requests.pop(req_id)
                token_ids = req_info.get("token_ids")
                if token_ids is not None:
                    token_ids = torch.tensor(token_ids, dtype=torch.long)
                
                hook_manager = get_hook_manager()
                hook_manager.set_request_id(req_id)
                save_path = hook_manager.save_hidden_states(token_ids)
                if save_path:
                    saved_paths.append(save_path)
        
        return None, None
    
    def update_state_after_alloc(
        self,
        request_id: str,
        new_block_ids: List[int],
        num_external_tokens: int,
    ) -> None:
        pass
    
    def build_partial_prefill_meta(
        self,
        request_id: str,
        block_ids: List[int],
        block_size: int,
        num_computed_tokens: int,
        num_tokens_to_compute: int,
    ) -> dict:
        return {}
    
    def request_finished(
        self,
        request_id: str,
        block_ids: List[int],
        is_prefill: bool,
        num_computed_tokens: int,
    ) -> Tuple[bool, Optional[dict]]:
        # Save hidden states when request finishes
        if request_id in self._pending_requests:
            req_info = self._pending_requests.pop(request_id)
            token_ids = req_info.get("token_ids")
            if token_ids is not None:
                token_ids = torch.tensor(token_ids, dtype=torch.long)
            
            hook_manager = get_hook_manager()
            hook_manager.set_request_id(request_id)
            save_path = hook_manager.save_hidden_states(token_ids)
            
            if save_path:
                return True, {"hidden_states_path": save_path}
        
        return True, None
