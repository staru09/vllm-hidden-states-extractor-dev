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

    # ── New GPU-resident tap connector ──
    if "HiddenStateTapConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "HiddenStateTapConnector",
            "vllm_hidden_states_extractor.hidden_tap",
            "HiddenStateTapConnector",
        )
        print("HiddenStateTapConnector registered")

    # Patch model classes to register tap hooks on initialization
    if os.environ.get("HIDDEN_TAP_ENABLED", "0") == "1":
        tap_layer = int(os.environ.get("HIDDEN_TAP_LAYER", "20"))
        _patch_model_for_tap(tap_layer)


def _patch_model_for_tap(tap_layer: int):
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
            _apply_tap_patch(model_cls, class_name, tap_layer)
        except (ImportError, AttributeError):
            # Model not available in this vLLM install, skip
            pass


def _apply_tap_patch(model_cls, class_name: str, tap_layer: int):
    """Apply the tap hook patch to a single model class."""
    from vllm_hidden_states_extractor.hidden_tap import register_tap_hooks
    from vllm_hidden_states_extractor.gpu_buffer import get_global_buffer

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *, vllm_config, prefix: str = "", **kwargs):
        original_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)

        buffer = get_global_buffer()
        try:
            handles = register_tap_hooks(self, tap_layer, buffer)
            self._tap_hook_handles = handles
            print(f"[HiddenTap] Registered tap on layer {tap_layer} of {class_name}")
        except Exception as e:
            print(f"[HiddenTap] Warning: Failed to register tap on {class_name}: {e}")

    model_cls.__init__ = patched_init
    print(f"[HiddenTap] {class_name} patched for tap hooks")
