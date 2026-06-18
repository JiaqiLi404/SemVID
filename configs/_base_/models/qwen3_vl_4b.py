model_name_or_path = "Qwen/Qwen3-VL-4B-Thinking"

eval_args = dict(
    evaluator_type="QwenTransformersEvaluator",
    type="BasicDatasetForQwen",
    eval_model_path=model_name_or_path,
    qwen_version=3,
    force_stop_thinking=True,
)

model_args = dict(model_name_or_path=model_name_or_path)
