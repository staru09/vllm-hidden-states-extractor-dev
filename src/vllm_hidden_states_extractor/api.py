# SPDX-License-Identifier: Apache-2.0
"""
FastAPI router for inspecting/consuming GPU-resident hidden states.

These routes run INSIDE the vLLM server process, giving them direct
access to the GPUBufferManager singleton.

Endpoints:
    GET  /hidden_states/inspect?handle=<id>   — shape, dtype, stats for one handle
    POST /hidden_states/consume               — batch consume handles, return stats, free slots
    GET  /hidden_states/buffer_stats           — buffer occupancy info
"""

from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from vllm_hidden_states_extractor.gpu_buffer import get_global_buffer

router = APIRouter(prefix="/hidden_states", tags=["hidden_states"])


def _tensor_stats(tensor: torch.Tensor) -> dict:
    """Compute basic stats for a tensor (handles bfloat16)."""
    t = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "memory_bytes": tensor.element_size() * tensor.numel(),
        "min": round(t.min().item(), 6),
        "max": round(t.max().item(), 6),
        "mean": round(t.mean().item(), 6),
        "std": round(t.std().item(), 6),
        "norm": round(t.norm().item(), 6),
    }


# ── GET /hidden_states/inspect ──

@router.get("/inspect")
def inspect_handle(handle: str = Query(..., description="Buffer handle ID")):
    """Inspect a single tensor by handle without freeing it."""
    buffer = get_global_buffer()
    tensor, metadata = buffer.get(handle)

    if tensor is None:
        raise HTTPException(status_code=404, detail=f"Handle '{handle}' not found or expired")

    return {
        "handle": handle,
        "metadata": metadata,
        "tensor": _tensor_stats(tensor),
    }


# ── POST /hidden_states/consume ──

class ConsumeRequest(BaseModel):
    handles: List[str]
    free_after: bool = True


@router.post("/consume")
def consume_handles(req: ConsumeRequest):
    """
    Batch-consume handles: return tensor info for each step, optionally free slots.

    Returns per-step shape/stats and an overall summary.
    """
    buffer = get_global_buffer()
    steps = []
    tensors_for_stack = []

    for i, handle in enumerate(req.handles):
        tensor, metadata = buffer.get(handle)
        if tensor is None:
            steps.append({
                "step": i,
                "handle": handle,
                "status": "not_found_or_expired",
            })
            continue

        step_info = {
            "step": i,
            "handle": handle,
            "status": "ok",
            "metadata": metadata,
            "tensor": _tensor_stats(tensor),
        }
        steps.append(step_info)
        tensors_for_stack.append(tensor)

        if req.free_after:
            buffer.free(handle)

    # Build summary
    summary = None
    if tensors_for_stack:
        stacked = torch.cat(tensors_for_stack, dim=0)
        summary = {
            "total_steps": len(tensors_for_stack),
            "stacked_shape": list(stacked.shape),
            "hidden_dim": stacked.shape[-1],
            "total_tokens": stacked.shape[0],
            "memory_bytes": stacked.element_size() * stacked.numel(),
            "prefill_shape": list(tensors_for_stack[0].shape),
            "decode_shape": list(tensors_for_stack[-1].shape),
            "overall_stats": _tensor_stats(stacked),
            "freed": req.free_after,
        }

    return {
        "steps": steps,
        "summary": summary,
    }


# ── GET /hidden_states/buffer_stats ──

@router.get("/buffer_stats")
def buffer_stats():
    """Get current GPU buffer occupancy info."""
    buffer = get_global_buffer()
    return buffer.get_stats()
