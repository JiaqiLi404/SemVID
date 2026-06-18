_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen3_vl_4b.py",
    "../_base_/methods/fastvid_qwen3.py",
    "../_base_/datasets/charades_sta.py",
]

training_args = dict(output_dir="outputs/fastvid/qwen3_vl_4b/charades_sta")
