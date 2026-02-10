# SPDX-License-Identifier: Apache-2.0
"""
GPU Buffer Manager for hidden states tap-out.

Manages a pool of GPU-resident tensors with handle-based access.
Tensors stay on GPU and are accessible via CUDA IPC handles
for cross-process consumption.

Key design:
- Pre-allocated ring buffer to avoid dynamic allocation during inference
- Handle-based access (integer IDs) returned to clients
- TTL-based expiry for unclaimed tensors
- CUDA IPC handle generation for cross-process access
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BufferSlot:
    """A single slot in the GPU buffer."""
    handle: str                          # Unique handle ID
    tensor: Optional[torch.Tensor]       # GPU tensor (None if free)
    created_at: float                    # Timestamp
    ttl: float                          # Time-to-live in seconds
    metadata: dict = field(default_factory=dict)  # Request metadata
    consumed: bool = False               # Whether the consumer read it

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def is_free(self) -> bool:
        return self.tensor is None or self.is_expired


class GPUBufferManager:
    """
    Manages GPU-resident tensors with handle-based access.
    
    Usage:
        manager = GPUBufferManager(max_slots=32, device="cuda:0")
        
        # Store a tensor (called from forward hook)
        handle = manager.store(tensor, metadata={"req_id": "abc"})
        
        # Retrieve a tensor (called from consumer)
        tensor, metadata = manager.get(handle)
        
        # Free the slot
        manager.free(handle)
    """

    def __init__(
        self,
        max_slots: int = 64,
        default_ttl: float = 30.0,
        device: str = "cuda:0",
    ):
        self._max_slots = max_slots
        self._default_ttl = default_ttl
        self._device = torch.device(device)
        self._slots: Dict[str, BufferSlot] = {}
        self._order: List[str] = []  # Order of insertion for eviction

    @property
    def num_active(self) -> int:
        return sum(1 for s in self._slots.values() if not s.is_free)

    @property
    def num_free(self) -> int:
        return self._max_slots - self.num_active

    def store(
        self,
        tensor: torch.Tensor,
        metadata: Optional[dict] = None,
        ttl: Optional[float] = None,
    ) -> str:
        """
        Store a GPU tensor in the buffer.
        
        Args:
            tensor: GPU tensor to store (will be detached, NOT moved to CPU)
            metadata: Optional metadata (request ID, layer info, etc.)
            ttl: Time-to-live in seconds (default: self._default_ttl)
            
        Returns:
            Handle string for retrieving the tensor
        """
        # Clean up expired slots first
        self._cleanup_expired()

        # Evict oldest if full
        if len(self._slots) >= self._max_slots:
            self._evict_oldest()

        handle = str(uuid.uuid4())[:12]
        
        # Detach but keep on GPU
        stored_tensor = tensor.detach().clone()

        slot = BufferSlot(
            handle=handle,
            tensor=stored_tensor,
            created_at=time.time(),
            ttl=ttl or self._default_ttl,
            metadata=metadata or {},
        )

        self._slots[handle] = slot
        self._order.append(handle)

        return handle

    def get(self, handle: str) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Retrieve a GPU tensor by handle.
        
        Returns:
            (tensor, metadata) or (None, {}) if not found/expired
        """
        slot = self._slots.get(handle)
        if slot is None or slot.is_free:
            return None, {}

        slot.consumed = True
        return slot.tensor, slot.metadata

    def get_ipc_handle(self, handle: str) -> Optional[bytes]:
        """
        Get a CUDA IPC handle for cross-process access.
        
        Returns:
            Serialized IPC handle bytes, or None if not found
        """
        slot = self._slots.get(handle)
        if slot is None or slot.is_free or slot.tensor is None:
            return None

        # Make tensor contiguous for IPC
        tensor = slot.tensor.contiguous()
        
        # Get CUDA IPC handle
        # This returns (storage, offset, size, stride, ...)
        ipc_data = tensor.storage()._share_cuda_()
        return ipc_data

    def free(self, handle: str) -> bool:
        """Free a buffer slot by handle."""
        slot = self._slots.pop(handle, None)
        if slot is not None:
            if handle in self._order:
                self._order.remove(handle)
            slot.tensor = None
            return True
        return False

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "max_slots": self._max_slots,
            "active_slots": self.num_active,
            "free_slots": self.num_free,
            "total_allocated": len(self._slots),
            "expired": sum(1 for s in self._slots.values() if s.is_expired),
            "consumed": sum(1 for s in self._slots.values() if s.consumed),
        }

    def _cleanup_expired(self):
        """Remove expired slots."""
        expired = [h for h, s in self._slots.items() if s.is_expired]
        for h in expired:
            self._slots.pop(h, None)
            if h in self._order:
                self._order.remove(h)

    def _evict_oldest(self):
        """Evict the oldest slot to make room."""
        if self._order:
            oldest = self._order.pop(0)
            slot = self._slots.pop(oldest, None)
            if slot:
                slot.tensor = None


# Global singleton instance
_global_buffer: Optional[GPUBufferManager] = None


def get_global_buffer() -> GPUBufferManager:
    """Get the global GPU buffer manager instance."""
    global _global_buffer
    if _global_buffer is None:
        _global_buffer = GPUBufferManager()
    return _global_buffer


def init_global_buffer(
    max_slots: int = 64,
    default_ttl: float = 30.0,
    device: str = "cuda:0",
) -> GPUBufferManager:
    """Initialize the global GPU buffer manager."""
    global _global_buffer
    _global_buffer = GPUBufferManager(
        max_slots=max_slots,
        default_ttl=default_ttl,
        device=device,
    )
    return _global_buffer
