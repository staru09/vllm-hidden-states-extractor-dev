import os
import functools

# Default layer indices to capture hidden states from
DEFAULT_LAYER_INDICES = [7, 14, 21, 28]


def register():
    from vllm import ModelRegistry
    from vllm.transformers_utils.configs.speculators.algos import (
        register_speculator,
        update_eagle3,
    )
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    @register_speculator("extract_hidden_states")
    def update_extract_hidden_states(config_dict: dict, vllm_config: dict) -> None:
        """
        This is a fake speculator that extracts hidden states from the model. It pretends to be an eagle3 speculator.
        """
        update_eagle3(config_dict, vllm_config)
        vllm_config["method"] = "eagle3"
        vllm_config["architectures"] = ["HiddenStatesExtractor"]

    print("HiddenStatesExtractor registered")
    if "HiddenStatesExtractor" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HiddenStatesExtractor",
            "vllm_hidden_states_extractor.model:HiddenStatesExtractor",
        )

    # Register the original connector (uses fake attention layers)
    if "ExampleHiddenStatesConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "ExampleHiddenStatesConnector",
            "vllm_hidden_states_extractor.connector",
            "ExampleHiddenStatesConnector",
        )

    # Register the new hook-based connector (uses forward hooks on real model)
    if "HookBasedHiddenStatesConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "HookBasedHiddenStatesConnector",
            "vllm_hidden_states_extractor.hook_connector",
            "HookBasedHiddenStatesConnector",
        )
        print("HookBasedHiddenStatesConnector registered")
    
    # Monkey-patch Qwen3 model to register hooks after initialization
    _patch_qwen3_model()


def _patch_qwen3_model():
    """
    Monkey-patch the Qwen3 model to register forward hooks after initialization.
    """
    try:
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm_hidden_states_extractor.model_wrapper import register_hooks_on_model
        
        # Store the original __init__
        original_init = Qwen3ForCausalLM.__init__
        
        @functools.wraps(original_init)
        def patched_init(self, *, vllm_config, prefix: str = "", **kwargs):
            # Call original init with proper keyword arguments
            original_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
            
            # Register hooks if environment variable is set
            if os.environ.get("ENABLE_HIDDEN_STATES_HOOKS", "0") == "1":
                layer_indices_str = os.environ.get("HIDDEN_STATES_LAYER_INDICES", "")
                if layer_indices_str:
                    layer_indices = [int(x) for x in layer_indices_str.split(",")]
                else:
                    layer_indices = DEFAULT_LAYER_INDICES
                
                try:
                    handles = register_hooks_on_model(self, layer_indices)
                    self._hidden_state_hook_handles = handles
                    print(f"Registered {len(handles)} hidden state hooks on Qwen3 model")
                except Exception as e:
                    print(f"Warning: Failed to register hooks: {e}")
        
        Qwen3ForCausalLM.__init__ = patched_init
        print("Qwen3ForCausalLM patched for hidden state hooks")
        
    except ImportError as e:
        print(f"Warning: Could not patch Qwen3 model: {e}")
    except Exception as e:
        print(f"Warning: Error patching Qwen3 model: {e}")
