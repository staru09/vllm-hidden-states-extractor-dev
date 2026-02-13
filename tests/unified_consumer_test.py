import argparse
import sys
import time
import requests
import json

def main():
    parser = argparse.ArgumentParser(description="Unified Hidden States Consumer Test")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--prompt", default="The future of AI is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens to generate")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--mode", choices=["standard", "realtime"], default="standard",
                        help="Mode to test: 'standard' (expects data in response) or 'realtime' (expects empty response but server logs)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Unified Consumer Test: {args.mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"  URL:        {args.url}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Model:      {args.model}")

    if args.mode == "realtime":
        print("\n  [!] REALTIME MODE NOTE:")
        print("      - Make sure vLLM was started with HIDDEN_STATES_REALTIME_CONSUMER=1")
        print("      - You should check the SERVER LOGS to see per-token 'step=' logs.")
        print("      - The response below will likely show handles but NO tensor stats (because they were freed).")

    print(f"\n  Sending request...")
    start_time = time.time()

    try:
        resp = requests.post(
            f"{args.url}/v1/completions",
            json={
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0, # Deterministic for testing
            },
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)

    latency = (time.time() - start_time) * 1000
    print(f"  ✓ Request completed in {latency:.2f}ms")

    # ── Parse Response ──
    try:
        result = resp.json()
    except json.JSONDecodeError:
        print(f"  ✗ Failed to decode JSON response: {resp.text}")
        sys.exit(1)

    if "choices" in result:
        text = result["choices"][0]["text"]
        print(f"\n  Generated Text: \"{text.strip()}\"")
    else:
        print(f"  ? Response format unexpected: {result}")
        text = ""

    kv_params = result.get("kv_transfer_params", {})
    handles = kv_params.get("hidden_states_handles", [])
    steps = kv_params.get("hidden_states_steps", []) # List of dicts with stats
    
    print(f"  Handles Received: {len(handles)}")

    # ── Verification Logic ──
    print(f"\n{'='*70}")
    print(f"  Validation: {args.mode.upper()} Mode")
    print(f"{'='*70}")

    if args.mode == "standard":
        # Standard Mode Expectation: We SHOULD receive per-step stats
        if steps:
            print(f"  ✓ PASSED: Received {len(steps)} step records as expected.")
            print_step_table(steps)
        else:
            print(f"  ✗ FAILED: Expected step stats but received none.")
            print("     Possibility: Real-time consumer is ON and freed the tensors?")

    elif args.mode == "realtime":
        # Real-Time Mode Expectation: We MIGHT receive empty steps because they were freed
        
        if len(steps) == 0 and len(handles) > 0:
             print(f"  ✓ PASSED: Received handles but NO step tensor data (Expected behavior).")
             print("     (The server-side consumer freed the tensors before the request finished.)")
        elif len(steps) > 0:
             print(f"  ? NOTE: Received {len(steps)} step records.")
             print("     This implies tensors were NOT freed immediately, or we were lucky.")
             print("     Check server logs to ensure [RealtimeConsumer] is running.")
             print_step_table(steps)
        else:
             print("  ? No handles or steps received.")

def print_step_table(steps):
    print(f"\n  {'Step':>5}  {'Type':>8}  {'Shape':>15}  {'Mean (Check)':>12}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*15}  {'-'*12}")
    
    for s in steps:
        stype = "prefill" if s.get("is_prefill", False) else "decode"
        shape = str(s.get("shape", "???"))
        mean = f"{s.get('mean', 0.0):.4f}"
        print(f"  {s.get('step', '?'):>5}  {stype:>8}  {shape:>15}  {mean:>12}")

if __name__ == "__main__":
    main()
