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
