_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen2_5_vl_7b.py",
    "../_base_/methods/prunevid_qwen2_5.py",
    "../_base_/datasets/charades_sta.py",
]

training_args = dict(output_dir="outputs/prunevid/qwen2_5_vl_7b/charades_sta")
