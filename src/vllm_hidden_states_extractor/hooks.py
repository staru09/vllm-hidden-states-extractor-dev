# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based hidden states extraction for vLLM.

This module provides a way to extract hidden states from specific layers
of a model during inference by registering forward hooks.
"""

import os
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Callable
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.logger import init_logger

logger = init_logger(__name__)


class HiddenStatesHookManager:
    """
    Manages forward hooks for extracting hidden states from model layers.
    
    This class registers hooks on specified layers and stores the captured
    hidden states for later retrieval and saving.
    """
    
    _instance: Optional["HiddenStatesHookManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.layer_indices: List[int] = []
        self.save_path: str = "/tmp/hidden_states"
        self.hidden_states_buffer: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.current_request_id: Optional[str] = None
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._lock = threading.Lock()
        
    def configure(self, layer_indices: List[int], save_path: str = "/tmp/hidden_states"):
        """Configure which layers to extract and where to save."""
        self.layer_indices = layer_indices
        self.save_path = save_path
        logger.info(f"HiddenStatesHookManager configured: layers={layer_indices}, path={save_path}")
        
    def set_request_id(self, request_id: str):
        """Set the current request ID for organizing saved files."""
        with self._lock:
            self.current_request_id = request_id
            self.hidden_states_buffer.clear()
    
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook function for a specific layer."""
        def hook(module, input, output):
            with self._lock:
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                # Clone and detach to avoid memory issues
                self.hidden_states_buffer[layer_idx].append(
                    hidden_states.detach().clone().cpu()
                )
        return hook
    
    def register_hooks(self, model: torch.nn.Module):
        """
        Register forward hooks on the specified layers of the model.
        
        Args:
            model: The transformer model (e.g., Qwen3ForCausalLM)
        """
        # Remove any existing hooks first
        self.remove_hooks()
        
        # Find the layers module
        layers = None
        for name, module in model.named_modules():
            if name.endswith('.layers') or name == 'model.layers':
                layers = module
                break
        
        if layers is None:
            logger.warning("Could not find model layers for hook registration")
            return
            
        # Register hooks on specified layer indices
        for idx in self.layer_indices:
            if idx < len(layers):
                layer = layers[idx]
                hook_handle = layer.register_forward_hook(self._create_hook(idx))
                self.hooks.append(hook_handle)
                logger.info(f"Registered hook on layer {idx}")
            else:
                logger.warning(f"Layer index {idx} out of range (model has {len(layers)} layers)")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def save_hidden_states(self, token_ids: Optional[torch.Tensor] = None) -> Optional[str]:
        """
        Save the captured hidden states to disk.
        
        Returns:
            Path to the saved file, or None if no states were captured.
        """
        with self._lock:
            if not self.hidden_states_buffer:
                logger.warning("No hidden states captured to save")
                return None
            
            request_id = self.current_request_id or "unknown"
            save_dir = Path(self.save_path) / request_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Stack hidden states from each layer
            tensors_to_save = {}
            for layer_idx, states_list in sorted(self.hidden_states_buffer.items()):
                if states_list:
                    # Concatenate all captures from this layer
                    stacked = torch.cat(states_list, dim=0)
                    tensors_to_save[f"layer_{layer_idx}"] = stacked
            
            if token_ids is not None:
                tensors_to_save["token_ids"] = token_ids.cpu()
            
            save_path = save_dir / "hidden_states.safetensors"
            save_file(tensors_to_save, str(save_path))
            
            logger.info(f"Saved hidden states to {save_path}")
            
            # Clear buffer after saving
            self.hidden_states_buffer.clear()
            
            return str(save_path)


def get_hook_manager() -> HiddenStatesHookManager:
    """Get the singleton HiddenStatesHookManager instance."""
    return HiddenStatesHookManager()
