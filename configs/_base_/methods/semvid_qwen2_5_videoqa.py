_base_ = ["semvid_qwen2_5.py"]

script_args = dict(
    model_hyper_parameters=dict(
        semantic_retention_ratio=0.24,
        semantic_stage1_topk_segments=0,
        semantic_obj_ratio=0.4,
        semantic_mmr_lambda=0.3,
        semantic_obj_anchor_num=1,
    )
)
