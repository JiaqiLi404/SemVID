_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen3_vl_4b.py",
    "../_base_/methods/baseline_qwen3.py",
    "../_base_/datasets/charades_sta.py",
]

script_args = dict(model_hyper_parameters=dict(pruning_method="random"))
training_args = dict(output_dir="outputs/baseline/random/qwen3_vl_4b/charades_sta")
