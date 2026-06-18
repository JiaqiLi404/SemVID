eval_args = dict(
    dataset_name="ActivityNet-Grounding",
    eval_data_path="data/ActivityNet/captions/val_2.json",
    eval_video_folder="data/ActivityNet/rgb_videos_15fps_short256",
    video_name_postfix=".mp4",
    prompt_task="temporal_grounding",
    is_shuffle=False,
)
