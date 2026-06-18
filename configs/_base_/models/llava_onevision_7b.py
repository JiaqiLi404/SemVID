model_name_or_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

eval_args = dict(
    evaluator_type="LLaVATransformersEvaluator",
    type="BasicDatasetForLlava",
    eval_model_path=model_name_or_path,
    force_stop_thinking=False,
    gpu_memory_utilization=0.75,
)

model_args = dict(model_name_or_path=model_name_or_path)
