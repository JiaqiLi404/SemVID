_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/llava_onevision_7b.py",
    "../_base_/methods/semvid_llava_videoqa.py",
    "../_base_/datasets/videomme.py",
]

eval_args = dict(sample_fps=None, max_num_frames=32, use_interleaved=False, longsize_resolution=None)
training_args = dict(output_dir="outputs/semvid/llava_onevision_7b/videomme")
