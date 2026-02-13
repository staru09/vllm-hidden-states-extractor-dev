"""
Consumer test script — inspect streaming hidden state tensors.

This reads per-step tensor shapes and stats directly from the
standard vLLM /v1/completions response (no custom API needed).

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
    steps = kv_params.get("hidden_states_steps", [])

    choice = result["choices"][0]
    usage = result.get("usage", {})

    print(f"  Generated: {choice['text'].strip()[:80]}...")
    print(f"  Prompt tokens: {usage.get('prompt_tokens', '?')}, "
          f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"  Handles received: {len(handles)}")

    if not handles:
        print("\n  ✗ No handles — nothing to inspect.")
        sys.exit(0)

    # ── Step 2: Per-step tensor table ──
    print(f"\n{'='*70}")
    print(f"  Per-Step Tensor Inspection")
    print(f"{'='*70}\n")

    if steps:
        print(f"  {'Step':>5}  {'Type':>8}  {'Shape':>25}  "
              f"{'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*25}  "
              f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for step in steps:
            step_type = "prefill" if step.get("is_prefill") else "decode"
            print(f"  {step['step']:>5}  {step_type:>8}  {str(step['shape']):>25}  "
                  f"{step['min']:>+10.4f}  {step['max']:>+10.4f}  "
                  f"{step['mean']:>+10.4f}  {step['std']:>10.4f}")
    else:
        print("  ⚠ No per-step data in response (older server version?)")
        for i, h in enumerate(handles):
            label = "prefill" if i == 0 else "decode"
            print(f"  Step {i:>3} ({label:>8}): handle={h}")

    # ── Step 3: Summary ──
    stacked_shape = kv_params.get("hidden_states_stacked_shape")
    if stacked_shape:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        print(f"  Total steps:           {kv_params.get('hidden_states_num_steps')}")
        print(f"  Stacked shape:         {stacked_shape}")
        print(f"  Hidden dimension:      {kv_params.get('hidden_states_hidden_dim')}")
        print(f"  Total tokens:          {kv_params.get('hidden_states_total_tokens')}")
        print(f"  Dtype:                 {kv_params.get('hidden_states_dtype')}")
        print(f"  Layer:                 {kv_params.get('hidden_states_layer')}")
        print(f"  Device:                {kv_params.get('hidden_states_device')}")

        if steps:
            prefill = steps[0]
            decode = steps[-1] if len(steps) > 1 else None
            print(f"\n  Prefill shape:         {prefill['shape']}")
            if decode:
                print(f"  Decode shape:          {decode['shape']}")

    print(f"\n{'='*70}")
    print(f"  Full kv_transfer_params:")
    print(f"{'='*70}")
    # Print without the verbose steps list
    compact = {k: v for k, v in kv_params.items() if k != "hidden_states_steps"}
    print(json.dumps(compact, indent=2))

    print(f"\n{'='*70}")
    print(f"  ✓ Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
