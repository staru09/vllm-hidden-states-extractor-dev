"""
Consumer test script — inspect streaming hidden state tensors via HTTP API.

Sends a generation request, then calls the /hidden_states/consume endpoint
to inspect each tensor's shape, stats, and get a stacked summary.

Requires: vLLM server started with run_server.py (which mounts the API).

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
    parser = argparse.ArgumentParser(description="Inspect streaming hidden states via API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--no-free", action="store_true",
                        help="Don't free buffer slots after consuming")
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

    # ── Step 2: Consume via API ──
    print(f"\n{'='*70}")
    print(f"  Consuming tensors via /hidden_states/consume")
    print(f"{'='*70}\n")

    try:
        consume_resp = requests.post(
            f"{args.url}/hidden_states/consume",
            json={
                "handles": handles,
                "free_after": not args.no_free,
            },
        )
        consume_resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ API error: {e}")
        print(f"    Make sure you started the server with run_server.py")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to {args.url}/hidden_states/consume")
        print(f"    Make sure you started the server with run_server.py")
        sys.exit(1)

    data = consume_resp.json()

    # ── Display per-step table ──
    steps = data.get("steps", [])
    print(f"  {'Step':>5}  {'Type':>8}  {'Shape':>25}  {'Dtype':>16}  "
          f"{'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*25}  {'-'*16}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for step in steps:
        if step.get("status") != "ok":
            print(f"  {step['step']:>5}  {'?':>8}  {'-- not found --':>25}")
            continue

        t = step["tensor"]
        meta = step.get("metadata", {})
        is_prefill = meta.get("is_prefill", step["step"] == 0)
        step_type = "prefill" if is_prefill else "decode"

        print(f"  {step['step']:>5}  {step_type:>8}  {str(t['shape']):>25}  "
              f"{t['dtype']:>16}  {t['min']:>+10.4f}  {t['max']:>+10.4f}  "
              f"{t['mean']:>+10.4f}  {t['std']:>10.4f}")

    # ── Summary ──
    summary = data.get("summary")
    if summary:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        print(f"  Total steps:           {summary['total_steps']}")
        print(f"  Prefill tensor shape:  {summary['prefill_shape']}")
        print(f"  Decode tensor shape:   {summary['decode_shape']}")
        print(f"  Stacked shape:         {summary['stacked_shape']}")
        print(f"  Hidden dimension:      {summary['hidden_dim']}")
        print(f"  Total tokens:          {summary['total_tokens']}")
        print(f"  GPU memory:            {summary['memory_bytes'] / 1024:.1f} KB")
        print(f"  Slots freed:           {summary['freed']}")

        overall = summary.get("overall_stats", {})
        if overall:
            print(f"\n  Overall stats:")
            print(f"    Min:  {overall['min']:+.6f}")
            print(f"    Max:  {overall['max']:+.6f}")
            print(f"    Mean: {overall['mean']:+.6f}")
            print(f"    Std:  {overall['std']:.6f}")
            print(f"    Norm: {overall['norm']:.6f}")

    # ── Buffer stats ──
    try:
        stats_resp = requests.get(f"{args.url}/hidden_states/buffer_stats")
        if stats_resp.ok:
            stats = stats_resp.json()
            print(f"\n  Buffer: {stats['active_slots']}/{stats['max_slots']} active, "
                  f"{stats['free_slots']} free, {stats.get('expired', 0)} expired")
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"  ✓ Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
