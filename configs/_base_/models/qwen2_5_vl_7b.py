model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

eval_args = dict(
    evaluator_type="QwenTransformersEvaluator",
    type="BasicDatasetForQwen",
    eval_model_path=model_name_or_path,
    qwen_version=2.5,
    force_stop_thinking=False,
    gpu_memory_utilization=0.80,
)

model_args = dict(model_name_or_path=model_name_or_path)
