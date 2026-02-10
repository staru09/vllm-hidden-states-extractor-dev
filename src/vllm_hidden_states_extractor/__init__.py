import os
import functools


def register():
    from vllm import ModelRegistry
    from vllm.transformers_utils.configs.speculators.algos import (
        register_speculator,
        update_eagle3,
    )
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    # ── Original speculative decoding approach ──
    @register_speculator("extract_hidden_states")
    def update_extract_hidden_states(config_dict: dict, vllm_config: dict) -> None:
        update_eagle3(config_dict, vllm_config)
        vllm_config["method"] = "eagle3"
        vllm_config["architectures"] = ["HiddenStatesExtractor"]

    print("HiddenStatesExtractor registered")
    if "HiddenStatesExtractor" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HiddenStatesExtractor",
            "vllm_hidden_states_extractor.model:HiddenStatesExtractor",
        )

    if "ExampleHiddenStatesConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "ExampleHiddenStatesConnector",
            "vllm_hidden_states_extractor.connector",
            "ExampleHiddenStatesConnector",
        )

    # ── New GPU-resident activation connector ──
    if "HiddenActivationsConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "HiddenActivationsConnector",
            "vllm_hidden_states_extractor.hidden_activations",
            "HiddenActivationsConnector",
        )
        print("HiddenActivationsConnector registered")

    # Patch model classes to register activation hooks on initialization
    if os.environ.get("HIDDEN_ACTIVATIONS_ENABLED", "0") == "1":
        activation_layer = int(os.environ.get("HIDDEN_ACTIVATIONS_LAYER", "20"))
        _patch_model_for_activations(activation_layer)


def _patch_model_for_activations(activation_layer: int):
    """
    Monkey-patch supported model classes to register a forward hook
    on the target layer after initialization.
    
    Supports: Llama, Qwen3 (add more as needed)
    """
    models_to_patch = [
        ("vllm.model_executor.models.llama", "LlamaForCausalLM"),
        ("vllm.model_executor.models.qwen3", "Qwen3ForCausalLM"),
    ]

    for module_path, class_name in models_to_patch:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            model_cls = getattr(mod, class_name)
            _apply_activation_patch(model_cls, class_name, activation_layer)
        except (ImportError, AttributeError):
            # Model not available in this vLLM install, skip
            pass


def _apply_activation_patch(model_cls, class_name: str, activation_layer: int):
    """Apply the activation hook patch to a single model class."""
    from vllm_hidden_states_extractor.hidden_activations import register_activation_hooks

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *, vllm_config, prefix: str = "", **kwargs):
        original_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)

        try:
            handles = register_activation_hooks(self, activation_layer)
            self._activation_hook_handles = handles
            print(f"[HiddenActivations] Registered hook on layer {activation_layer} of {class_name}")
        except Exception as e:
            print(f"[HiddenActivations] Warning: Failed to register hook on {class_name}: {e}")

    model_cls.__init__ = patched_init
    print(f"[HiddenActivations] {class_name} patched for activation hooks")
