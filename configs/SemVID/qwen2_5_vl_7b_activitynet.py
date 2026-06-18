_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen2_5_vl_7b.py",
    "../_base_/methods/semvid_qwen2_5.py",
    "../_base_/datasets/activitynet_grounding.py",
]

training_args = dict(output_dir="outputs/semvid/qwen2_5_vl_7b/activitynet_grounding")
