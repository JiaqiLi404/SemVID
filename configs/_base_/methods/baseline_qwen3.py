script_args = dict(
    model_handler="modeling_qwen3_vl_baseline.Qwen3VLForConditionalGeneration",
    model_hyper_parameters=dict(
        retention_ratio=0.125,
        pruning_method="query_relevance",
    ),
)
