eval_args = dict(
    prompt_task="temporal_grounding",
    prompt_cls=dict(type="Qwen3Prompts", thinking=False),
    video_name_only=False,
    is_shuffle=False,
    eval_batch_size=1,
    eval_gpus=1,
    eval_max_new_tokens=200,
    gpu_memory_utilization=0.85,
    max_model_len=36864,
)

training_args = dict(output_dir="outputs/semvid")
