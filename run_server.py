"""
Custom vLLM server launcher with hidden states API endpoints.

This starts a standard vLLM OpenAI-compatible server AND mounts the
/hidden_states/* API endpoints for inspecting GPU-resident tensors.

Usage (drop-in replacement for `vllm serve`):
    python run_server.py --model Qwen/Qwen3-8B --enforce-eager \
        --kv-transfer-config '{
            "kv_connector": "HiddenActivationsConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"activation_layer": 20}
        }'

All standard vLLM serve arguments are supported.
"""

import sys
import asyncio


def main():
    # Import vLLM's server machinery
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        make_arg_parser,
        run_server,
    )

    # Build the vLLM arg parser and parse our CLI args
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()

    # Patch: inject our custom router via the app creation hook
    # vLLM's run_server creates the FastAPI app internally,
    # so we hook into it by monkey-patching the build_app function
    import vllm.entrypoints.openai.api_server as api_module

    original_build_app = api_module.build_app

    def patched_build_app(args):
        app = original_build_app(args)

        # Mount our hidden states API
        from vllm_hidden_states_extractor.api import router
        app.include_router(router)
        print("[HiddenStates] API endpoints mounted:")
        print("[HiddenStates]   GET  /hidden_states/inspect?handle=<id>")
        print("[HiddenStates]   POST /hidden_states/consume")
        print("[HiddenStates]   GET  /hidden_states/buffer_stats")

        return app

    api_module.build_app = patched_build_app

    # Run the server (standard vLLM flow)
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
