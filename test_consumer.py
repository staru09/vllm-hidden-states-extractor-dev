"""
Consumer test script — inspect streaming hidden state tensors.

Sends a request to vLLM, gets the handles back, then either:
  1. If running IN the vLLM worker process: iterates tensors and shows shapes/stats
  2. If running as a SEPARATE process: shows handle metadata from the API response

For option 1, integrate this into a custom vLLM endpoint or use the
vLLM entrypoints API to co-locate the consumer.

Usage:
    python test_consumer.py
    python test_consumer.py --prompt "Explain gravity" --max-tokens 20
    python test_consumer.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import sys

import requests


def fetch_hidden_states(url, prompt, max_tokens, model):
    """Send a completion request and return the response."""
    resp = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    return resp.json()


def try_consume_in_process(handles):
    """
    Try to consume tensors from the in-process GPU buffer.
    Returns (tensors_list, metadata_list) or (None, None) if not available.
    """
    try:
        from vllm_hidden_states_extractor.consumer import StreamingHiddenStatesConsumer
        consumer = StreamingHiddenStatesConsumer()

        # Test first handle to see if buffer is populated
        tensor, meta = consumer._buffer.get(handles[0])
        if tensor is None:
            return None, None

        # First one worked — collect all
        tensors = [tensor]
        metas = [meta]
        for handle in handles[1:]:
            t, m = consumer._buffer.get(handle)
            if t is not None:
                tensors.append(t)
                metas.append(m)

        return tensors, metas
    except ImportError:
        return None, None


def display_tensor_table(tensors, metas):
    """Display per-step tensor info in a table."""
    import torch

    print(f"  {'Step':>5}  {'Type':>8}  {'Shape':>25}  {'Dtype':>16}  "
          f"{'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*25}  {'-'*16}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for i, (tensor, meta) in enumerate(zip(tensors, metas)):
        is_prefill = meta.get("is_prefill", i == 0)
        step_type = "prefill" if is_prefill else "decode"

        t_float = tensor.float()
        t_min = t_float.min().item()
        t_max = t_float.max().item()
        t_mean = t_float.mean().item()
        t_std = t_float.std().item()

        print(f"  {i:>5}  {step_type:>8}  {str(list(tensor.shape)):>25}  "
              f"{str(tensor.dtype):>16}  {t_min:>+10.4f}  {t_max:>+10.4f}  "
              f"{t_mean:>+10.4f}  {t_std:>10.4f}")

    # Summary
    stacked = torch.cat(tensors, dim=0)

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  Total steps:           {len(tensors)}")
    print(f"  Prefill tensor shape:  {list(tensors[0].shape)}")
    print(f"  Decode tensor shape:   {list(tensors[-1].shape)}")
    print(f"  Stacked shape:         {list(stacked.shape)}")
    print(f"  Hidden dimension:      {stacked.shape[-1]}")
    print(f"  Dtype:                 {stacked.dtype}")
    print(f"  Device:                {stacked.device}")
    print(f"  GPU memory (stacked):  {stacked.element_size() * stacked.numel() / 1024:.1f} KB")

    s_float = stacked.float()
    print(f"\n  Overall stats:")
    print(f"    Min:  {s_float.min().item():+.6f}")
    print(f"    Max:  {s_float.max().item():+.6f}")
    print(f"    Mean: {s_float.mean().item():+.6f}")
    print(f"    Std:  {s_float.std().item():.6f}")


def display_handle_info(handles, kv_params):
    """Display handle metadata when tensors aren't directly accessible."""
    print(f"  ⚠ Running in a SEPARATE process — cannot access GPU buffer directly.")
    print(f"    The handles are only valid inside the vLLM worker process.\n")
    print(f"  Metadata from API response:")
    print(f"    Steps:  {kv_params.get('hidden_states_num_steps', len(handles))}")
    print(f"    Dtype:  {kv_params.get('hidden_states_dtype', 'N/A')}")
    print(f"    Layer:  {kv_params.get('hidden_states_layer', 'N/A')}")
    print(f"    Device: {kv_params.get('hidden_states_device', 'N/A')}")
    print(f"\n  Handle IDs ({len(handles)} total):")

    # Show first few and last few
    show_n = 5
    for i, h in enumerate(handles[:show_n]):
        label = "prefill" if i == 0 else f"decode"
        print(f"    Step {i:>3} ({label:>8}): {h}")
    if len(handles) > show_n * 2:
        print(f"    {'...':>28}  ({len(handles) - show_n * 2} more)")
    for i, h in enumerate(handles[-show_n:], start=len(handles) - show_n):
        print(f"    Step {i:>3} ({'decode':>8}): {h}")

    print(f"\n  To access tensors, consume them IN the vLLM worker process:")
    print(f"    from vllm_hidden_states_extractor.consumer import StreamingHiddenStatesConsumer")
    print(f"    consumer = StreamingHiddenStatesConsumer()")
    print(f"    for tensor, meta in consumer.iter_steps(handles):")
    print(f"        print(tensor.shape)  # prefill: [seq_len, H], decode: [1, H]")


def main():
    parser = argparse.ArgumentParser(description="Inspect streaming hidden states")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    # ── Step 1: Generate ──
    print(f"\n{'='*70}")
    print(f"  Streaming Hidden States Consumer Test")
    print(f"{'='*70}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model:      {args.model}\n")

    try:
        result = fetch_hidden_states(args.url, args.prompt, args.max_tokens, args.model)
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to {args.url}")
        sys.exit(1)

    kv_params = result.get("kv_transfer_params", {})
    handles = kv_params.get("hidden_states_handles", [])

    choice = result["choices"][0]
    usage = result.get("usage", {})

    print(f"  Generated: {choice['text'].strip()[:80]}...")
    print(f"  Prompt tokens: {usage.get('prompt_tokens', '?')}, "
          f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"  Handles received: {len(handles)}")

    if not handles:
        print("\n  ✗ No handles — nothing to consume.")
        sys.exit(0)

    # ── Step 2: Try to consume tensors ──
    print(f"\n{'='*70}")
    print(f"  Per-Step Tensor Inspection")
    print(f"{'='*70}")

    tensors, metas = try_consume_in_process(handles)

    if tensors:
        display_tensor_table(tensors, metas)
    else:
        display_handle_info(handles, kv_params)

    print(f"\n{'='*70}")
    print(f"  ✓ Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
