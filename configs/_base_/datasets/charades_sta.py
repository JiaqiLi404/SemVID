eval_args = dict(
    dataset_name="Charades-STA",
    eval_data_path="data/Charades/sta_annotation/charades_sta_test.txt",
    eval_video_folder="data/Charades/rgb_videos_30fps_480",
    video_name_postfix=".mp4",
    prompt_task="temporal_grounding",
    is_shuffle=False,
)
