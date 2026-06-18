model_name_or_path = "Qwen/Qwen3-VL-8B-Thinking"

eval_args = dict(
    evaluator_type="QwenTransformersEvaluator",
    type="BasicDatasetForQwen",
    eval_model_path=model_name_or_path,
    qwen_version=3,
    force_stop_thinking=True,
    gpu_memory_utilization=0.80,
)

model_args = dict(model_name_or_path=model_name_or_path)
