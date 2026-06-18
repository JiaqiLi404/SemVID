_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen3_vl_8b.py",
    "../_base_/methods/semvid_qwen3.py",
    "../_base_/datasets/charades_sta.py",
]

training_args = dict(output_dir="outputs/semvid/qwen3_vl_8b/charades_sta")
