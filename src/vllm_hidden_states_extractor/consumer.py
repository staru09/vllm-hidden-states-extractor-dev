"""
Consumer client for GPU-resident hidden states.

This module runs as a co-process on the same GPU and reads
hidden states tensors directly from GPU memory using the buffer handle.

Usage as script:
    python -m vllm_hidden_states_extractor.consumer --handle <handle_id>

Usage as library:
    from vllm_hidden_states_extractor.consumer import HiddenStatesConsumer
    
    consumer = HiddenStatesConsumer()
    tensor = consumer.get("handle_id")
    output = my_model(tensor)
"""

import argparse
import sys
from typing import Optional, Tuple

import torch


class HiddenStatesConsumer:
    """
    Client for reading GPU-resident hidden states from the buffer.
    
    This must run in the SAME process as the vLLM worker (e.g., via a
    co-located service or by importing into a custom endpoint).
    
    For cross-process access, use HiddenStatesIPCConsumer instead.
    """

    def __init__(self):
        from vllm_hidden_states_extractor.gpu_buffer import get_global_buffer
        self._buffer = get_global_buffer()

    def get(self, handle: str) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Get a GPU tensor by handle.
        
        Returns:
            (tensor, metadata) - tensor is on GPU, metadata has shape/dtype/layer info
        """
        return self._buffer.get(handle)

    def free(self, handle: str) -> bool:
        """Free a buffer slot after consumption."""
        return self._buffer.free(handle)

    def stats(self) -> dict:
        """Get buffer statistics."""
        return self._buffer.get_stats()


class HiddenStatesIPCConsumer:
    """
    Cross-process consumer using CUDA IPC.
    
    This can run in a DIFFERENT process on the same machine.
    It connects to the GPU buffer via CUDA IPC handles.
    
    NOTE: This requires the producer process to share IPC handles
    via a side channel (file, socket, shared memory, etc.)
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = torch.device(device)

    def open_from_ipc_handle(self, ipc_data) -> torch.Tensor:
        """
        Reconstruct a GPU tensor from CUDA IPC handle data.
        
        Args:
            ipc_data: The IPC data from GPUBufferManager.get_ipc_handle()
            
        Returns:
            GPU tensor (shared memory with producer)
        """
        storage = torch.cuda.StorageFromIPCHandle(self._device, *ipc_data)
        return torch.tensor(storage)


def main():
    """CLI for inspecting buffer state (must run in same process)."""
    parser = argparse.ArgumentParser(description="Hidden states consumer")
    parser.add_argument("--handle", help="Buffer handle to read")
    parser.add_argument("--stats", action="store_true", help="Show buffer stats")
    args = parser.parse_args()

    consumer = HiddenStatesConsumer()

    if args.stats:
        print("Buffer stats:", consumer.stats())
        return

    if args.handle:
        tensor, metadata = consumer.get(args.handle)
        if tensor is not None:
            print(f"Handle:   {args.handle}")
            print(f"Shape:    {tensor.shape}")
            print(f"Dtype:    {tensor.dtype}")
            print(f"Device:   {tensor.device}")
            print(f"Metadata: {metadata}")
        else:
            print(f"Handle {args.handle} not found or expired")
        return

    print("Usage: --handle <id> or --stats")


if __name__ == "__main__":
    main()
