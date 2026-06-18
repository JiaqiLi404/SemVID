script_args = dict(
    model_handler="modeling_qwen3_vl_semvid.Qwen3VLForConditionalGenerationSemVID",
    model_hyper_parameters=dict(
        enable_semantic_prune=True,
        fastvid_DySeg_c=0,
        fastvid_DySeg_tau=0,
        semantic_retention_ratio=0.125,
        semantic_stage1_topk_segments=0,
        semantic_stage1_smooth_win=1,
        semantic_frame_weight_alpha=0.7,
        semantic_obj_ratio=0.6,
        semantic_mmr_lambda=0.9,
        semantic_min_tokens_per_frame=0,
        semantic_motion_query_beta=0.5,
        ablation_uniform_allocation=False,
        ablation_use_semantic_selection="semantic",
    ),
)
