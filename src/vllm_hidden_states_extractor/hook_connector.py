# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based hidden states connector for vLLM.

This connector saves hidden states by integrating with vLLM's KV connector system.
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

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
        )


@dataclass
class HookBasedConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(req_id, filename, token_ids, block_ids, block_size)
        )


class HookBasedHiddenStatesConnector(KVConnectorBase_V1):
    """
    A KV connector that saves hidden states from any layer.
    
    This is a simplified version that works with standard models.
    It captures hidden states from the KV cache layers specified.
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
        
        # Map layer index to layer name for filtering
        self._target_layers = set()
        self._request_filenames: dict[str, str] = {}
        self._layer_data: dict[str, dict[int, torch.Tensor]] = {}  # req_id -> {layer_idx: tensor}
        
        logger.info(f"HookBasedHiddenStatesConnector initialized: path={self._storage_path}, layers={self._layer_indices}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches and identify target layers."""
        layer_names = list(kv_caches.keys())
        for name in layer_names:
            # Extract layer index from name like "model.layers.7.self_attn"
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx in self._layer_indices:
                            self._target_layers.add(name)
                            logger.info(f"Will capture hidden states from layer: {name}")
                    except ValueError:
                        pass
        logger.info(f"Found {len(self._target_layers)} target layers to capture")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading KV cache (not used for store-only connector)."""
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load (not used for store-only connector)."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Save KV cache from specified layers."""
        if layer_name not in self._target_layers:
            return
            
        # Extract layer index
        layer_idx = None
        parts = layer_name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        
        if layer_idx is None:
            return
            
        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, HookBasedConnectorMetadata):
            return
            
        os.makedirs(self._storage_path, exist_ok=True)
        
        for request in connector_metadata.requests:
            # Extract KV for this request
            num_tokens = request.token_ids.shape[0]
            num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
            
            # Reshape and extract
            slot_mapping = request.slot_mapping[:num_tokens]
            padded_kv = kv_layer.reshape(2, num_pages * page_size, -1)[
                :, slot_mapping, ...
            ]
            
            # Store for this request
            if request.req_id not in self._layer_data:
                self._layer_data[request.req_id] = {}
            self._layer_data[request.req_id][layer_idx] = padded_kv.detach().cpu()

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
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename
        
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes - save collected hidden states."""
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        
        # Save any collected layer data
        if req_id in self._layer_data:
            layer_data = self._layer_data.pop(req_id)
            if layer_data and req_filename:
                tensors = {}
                for layer_idx, tensor in sorted(layer_data.items()):
                    tensors[f"layer_{layer_idx}"] = tensor
                save_file(tensors, req_filename)
                logger.info(f"Saved hidden states from {len(layer_data)} layers to {req_filename}")
        
        return False, {"hidden_states_path": req_filename}

    def clear_connector_metadata(self):
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
