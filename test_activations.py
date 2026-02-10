"""
Test script for the GPU-resident hidden states extraction system.

Sends a request to the vLLM server and shows:
1. Generated text output
2. Buffer handle for GPU-resident hidden states
3. Shape, dtype, and layer info of the captured states

Usage:
    python test_activations.py
    python test_activations.py --prompt "Explain gravity" --max-tokens 100
    python test_activations.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Test hidden state extraction")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Hidden Activations Test")
    print(f"{'='*60}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model:      {args.model}")
    print()

    # Send request
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
        print(f"    Start with:")
        print(f"    HIDDEN_ACTIVATIONS_ENABLED=1 vllm serve {args.model} --enforce-eager \\")
        print(f'      --kv-transfer-config \'{{')
        print(f'        "kv_connector": "HiddenActivationsConnector",')
        print(f'        "kv_role": "kv_producer",')
        print(f'        "kv_connector_extra_config": {{"activation_layer": 20}}')
        print(f"      }}'")
        sys.exit(1)

    result = resp.json()

    # Display output
    choice = result["choices"][0]
    usage = result.get("usage", {})
    kv_params = result.get("kv_transfer_params", {})

    print(f"{'='*60}")
    print(f"  Generated Output")
    print(f"{'='*60}")
    print(f"  {choice['text'].strip()}")
    print()
    print(f"  Finish reason:     {choice['finish_reason']}")
    print(f"  Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
    print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"  Hidden States Info (GPU-Resident)")
    print(f"{'='*60}")

    if kv_params.get("hidden_states_handle"):
        print(f"  ✓ Buffer handle: {kv_params['hidden_states_handle']}")
        print(f"  ✓ Shape:         {kv_params.get('hidden_states_shape', 'N/A')}")
        print(f"  ✓ Dtype:         {kv_params.get('hidden_states_dtype', 'N/A')}")
        print(f"  ✓ Layer:         {kv_params.get('hidden_states_layer', 'N/A')}")
        print(f"  ✓ Device:        {kv_params.get('hidden_states_device', 'N/A')}")
        print()
        print(f"  The tensor is still on GPU. To consume it:")
        print(f"    from vllm_hidden_states_extractor.consumer import HiddenStatesConsumer")
        print(f"    consumer = HiddenStatesConsumer()")
        print(f"    tensor, meta = consumer.get('{kv_params['hidden_states_handle']}')")
    else:
        print(f"  ✗ No hidden states captured")
        print(f"    kv_transfer_params: {json.dumps(kv_params, indent=2)}")

    print(f"\n{'='*60}")
    print(f"  Full kv_transfer_params:")
    print(f"{'='*60}")
    print(json.dumps(kv_params, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✓ Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
