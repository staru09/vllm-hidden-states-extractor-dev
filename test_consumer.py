"""
Consumer test script — inspect streaming hidden state tensors.

Sends a request to vLLM, gets the handles back, then iterates
through every step to show shape, dtype, stats, and prefill vs decode.

Usage:
    python test_consumer.py
    python test_consumer.py --prompt "Explain gravity" --max-tokens 20
    python test_consumer.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import sys

import requests


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
        resp = requests.post(
            f"{args.url}/v1/completions",
            json={
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
            },
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to {args.url}")
        sys.exit(1)

    result = resp.json()
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

    # ── Step 2: Consume tensors ──
    # Import here so it only runs in the vLLM worker process
    # For cross-process, you'd need IPC handles instead
    try:
        from vllm_hidden_states_extractor.consumer import StreamingHiddenStatesConsumer
    except ImportError:
        print("\n  ⚠ Cannot import consumer (not in vLLM worker process).")
        print("  Showing handle info from API response only.\n")
        for i, h in enumerate(handles):
            label = "prefill" if i == 0 else f"decode {i}"
            print(f"  Step {i:>3} ({label:>10}): handle={h}")
        sys.exit(0)

    consumer = StreamingHiddenStatesConsumer()

    print(f"\n{'='*70}")
    print(f"  Per-Step Tensor Inspection")
    print(f"{'='*70}")
    print(f"  {'Step':>5}  {'Type':>8}  {'Shape':>20}  {'Dtype':>16}  "
          f"{'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*20}  {'-'*16}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    all_tensors = []
    for i, (tensor, meta) in enumerate(consumer.iter_steps(handles, auto_free=False)):
        is_prefill = meta.get("is_prefill", False)
        step_type = "prefill" if is_prefill else "decode"

        # Compute stats (cast to float for bfloat16 compatibility)
        t_float = tensor.float()
        t_min = t_float.min().item()
        t_max = t_float.max().item()
        t_mean = t_float.mean().item()
        t_std = t_float.std().item()

        print(f"  {i:>5}  {step_type:>8}  {str(tensor.shape):>20}  "
              f"{str(tensor.dtype):>16}  {t_min:>+10.4f}  {t_max:>+10.4f}  "
              f"{t_mean:>+10.4f}  {t_std:>10.4f}")

        all_tensors.append(tensor)

    # ── Step 3: Summary ──
    import torch
    stacked = torch.cat(all_tensors, dim=0)

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  Total steps:           {len(all_tensors)}")
    print(f"  Prefill tensor shape:  {all_tensors[0].shape}")
    print(f"  Decode tensor shape:   {all_tensors[-1].shape}")
    print(f"  Stacked shape:         {stacked.shape}")
    print(f"  Hidden dimension:      {stacked.shape[-1]}")
    print(f"  Dtype:                 {stacked.dtype}")
    print(f"  Device:                {stacked.device}")
    print(f"  GPU memory (stacked):  {stacked.element_size() * stacked.numel() / 1024:.1f} KB")

    # Overall stats
    s_float = stacked.float()
    print(f"\n  Overall stats:")
    print(f"    Min:  {s_float.min().item():+.6f}")
    print(f"    Max:  {s_float.max().item():+.6f}")
    print(f"    Mean: {s_float.mean().item():+.6f}")
    print(f"    Std:  {s_float.std().item():.6f}")

    # Now free all handles
    for handle in handles:
        consumer._buffer.free(handle)
    print(f"\n  ✓ All {len(handles)} buffer slots freed.")

    print(f"\n{'='*70}")
    print(f"  ✓ Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
