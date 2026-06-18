script_args = dict(
    model_handler="modeling_llava_onevision_semvid.LlavaOnevisionForConditionalGeneration",
    model_hyper_parameters=dict(
        enable_semantic_prune=True,
        fastvid_DySeg_c=0,
        fastvid_DySeg_tau=0,
        semantic_retention_ratio=0.25,
        semantic_stage1_topk_segments=0,
        semantic_stage1_smooth_win=3,
        semantic_frame_weight_alpha=0.7,
        semantic_obj_ratio=0.3,
        semantic_mmr_lambda=0.5,
        semantic_min_tokens_per_frame=5,
        semantic_motion_query_beta=0.1,
    ),
)
