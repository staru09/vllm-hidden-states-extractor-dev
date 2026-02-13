"""
Test script for hidden states extraction via vLLM.

Sends a prompt to the vLLM server, displays the generated text,
and inspects the saved hidden states safetensor file.

Usage:
    python test_hidden_states.py
    python test_hidden_states.py --prompt "Explain quantum computing"
    python test_hidden_states.py --max-tokens 100
    python test_hidden_states.py --url http://localhost:8000
"""

import argparse
import json
import sys
import os

import requests


def send_prompt(url: str, prompt: str, max_tokens: int, model: str) -> dict:
    """Send a completion request to the vLLM server."""
    endpoint = f"{url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    
    print(f"{'='*60}")
    print(f"  Sending request to {endpoint}")
    print(f"{'='*60}")
    print(f"  Prompt:     {prompt}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Model:      {model}")
    print()
    
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    return response.json()


def display_response(result: dict):
    """Display the model's generated output."""
    print(f"{'='*60}")
    print(f"  Model Output")
    print(f"{'='*60}")
    
    choice = result["choices"][0]
    print(f"  Generated text:\n")
    print(f"    {choice['text'].strip()}")
    print()
    print(f"  Finish reason: {choice['finish_reason']}")
    
    usage = result.get("usage", {})
    print(f"  Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
    print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
    print(f"  Total tokens:      {usage.get('total_tokens', 'N/A')}")
    print()
    
    # Extract hidden states path from kv_transfer_params
    kv_params = result.get("kv_transfer_params", {})
    hidden_states_path = kv_params.get("hidden_states_path")
    
    if hidden_states_path:
        print(f"  Hidden states path: {hidden_states_path}")
    else:
        print("  ⚠ No hidden states path found in response")
    
    print()
    return hidden_states_path


def inspect_safetensor(filepath: str):
    """Inspect a saved safetensor file with hidden states."""
    from safetensors.torch import load_file
    
    print(f"{'='*60}")
    print(f"  Hidden States Details")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"  File: {filepath}")
    print(f"  Size: {file_size / 1024:.1f} KB ({file_size:,} bytes)")
    print()
    
    data = load_file(filepath)
    
    # Separate token_ids from layer data
    token_ids = data.pop("token_ids", None)
    
    if token_ids is not None:
        print(f"  Token IDs")
        print(f"  ─────────")
        print(f"    Shape: {token_ids.shape}")
        print(f"    Dtype: {token_ids.dtype}")
        print(f"    Values: {token_ids.tolist()}")
        print()
    
    # Sort layer keys numerically
    layer_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    
    print(f"  Hidden State Layers ({len(layer_keys)} captured)")
    print(f"  ──────────────────────────────────────")
    
    for key in layer_keys:
        tensor = data[key]
        layer_num = key.split("_")[1]
        print(f"    Layer {layer_num:>3}: shape={str(tensor.shape):>20}  dtype={tensor.dtype}  "
              f"min={tensor.float().min().item():+.4f}  max={tensor.float().max().item():+.4f}  "
              f"mean={tensor.float().mean().item():+.4f}")
    
    print()
    
    # Summary
    if layer_keys:
        sample = data[layer_keys[0]]
        total_captures, hidden_dim = sample.shape
        print(f"  Summary")
        print(f"  ───────")
        print(f"    Layers captured:     {len(layer_keys)}")
        print(f"    Total captures/layer: {total_captures}")
        print(f"    Hidden dimension:    {hidden_dim}")
        print(f"    Total parameters:    {sum(t.numel() for t in data.values()):,}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test hidden states extraction from vLLM")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    args = parser.parse_args()
    
    print()
    
    # Step 1: Send prompt
    try:
        result = send_prompt(args.url, args.prompt, args.max_tokens, args.model)
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Could not connect to {args.url}")
        print(f"    Make sure the vLLM server is running with:")
        print(f'    ENABLE_HIDDEN_STATES_HOOKS=1 vllm serve {args.model} --enforce-eager \\')
        print(f'      --kv-transfer-config \'{{...}}\'')
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ Server error: {e}")
        sys.exit(1)
    
    # Step 2: Display response
    hidden_states_path = display_response(result)
    
    # Step 3: Inspect safetensor
    if hidden_states_path:
        inspect_safetensor(hidden_states_path)
    
    print(f"{'='*60}")
    print(f"  ✓ Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
