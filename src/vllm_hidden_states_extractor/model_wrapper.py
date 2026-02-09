# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Qwen3 wrapper model that adds forward hooks for hidden states extraction.
"""

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import save_file

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global storage for captured hidden states
_captured_hidden_states: Dict[int, List[torch.Tensor]] = OrderedDict()
_capture_lock = threading.Lock()
_current_request_info: Dict[str, Any] = {}


def get_captured_states() -> Dict[int, List[torch.Tensor]]:
    """Get the captured hidden states."""
    return _captured_hidden_states


def clear_captured_states():
    """Clear the captured hidden states."""
    global _captured_hidden_states
    with _capture_lock:
        _captured_hidden_states.clear()


def set_current_request(req_id: str, token_ids: List[int], save_path: str):
    """Set the current request info."""
    global _current_request_info
    _current_request_info = {
        "req_id": req_id,
        "token_ids": token_ids,
        "save_path": save_path,
    }


def save_captured_states_for_request() -> Optional[str]:
    """Save captured states for the current request."""
    global _captured_hidden_states, _current_request_info
    
    with _capture_lock:
        if not _captured_hidden_states:
            logger.warning("No hidden states captured")
            return None
        
        req_info = _current_request_info
        if not req_info:
            logger.warning("No request info set")
            return None
        
        save_dir = req_info.get("save_path", "/tmp/hidden_states")
        req_id = req_info.get("req_id", "unknown")
        token_ids = req_info.get("token_ids", [])
        
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{req_id}.safetensors")
        
        tensors = {}
        for layer_idx, states_list in sorted(_captured_hidden_states.items()):
            if states_list:
                stacked = torch.cat(states_list, dim=0)
                tensors[f"layer_{layer_idx}"] = stacked
        
        if token_ids:
            tensors["token_ids"] = torch.tensor(token_ids, dtype=torch.long)
        
        if tensors:
            save_file(tensors, filename)
            logger.info(f"Saved hidden states from {len(_captured_hidden_states)} layers to {filename}")
        
        _captured_hidden_states.clear()
        return filename


def create_layer_hook(layer_idx: int):
    """Create a forward hook for a specific layer."""
    def hook(module, input, output):
        with _capture_lock:
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if layer_idx not in _captured_hidden_states:
                _captured_hidden_states[layer_idx] = []
            
            # Clone and detach to CPU
            _captured_hidden_states[layer_idx].append(
                hidden_states.detach().clone().cpu()
            )
            
    return hook


def register_hooks_on_model(model: torch.nn.Module, layer_indices: List[int]) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on the specified layers of a model.
    
    Args:
        model: The model to register hooks on
        layer_indices: List of layer indices to hook
        
    Returns:
        List of hook handles for cleanup
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
        # Try alternative paths
        for name, module in model.named_modules():
            if 'layers' in name and isinstance(module, torch.nn.ModuleList):
                layers = module
                logger.info(f"Found layers at: {name}")
                break
    
    if layers is None:
        logger.warning("Could not find model layers for hook registration")
        return handles
    
    # Register hooks
    for idx in layer_indices:
        if idx < len(layers):
            layer = layers[idx]
            hook_handle = layer.register_forward_hook(create_layer_hook(idx))
            handles.append(hook_handle)
            logger.info(f"Registered forward hook on layer {idx}")
        else:
            logger.warning(f"Layer index {idx} out of range (model has {len(layers)} layers)")
    
    return handles
