# Data preparation

Open Visual-Pruning Suite does not redistribute videos, annotations, or subtitles. Download each benchmark from its
official source and follow its license and access terms.

## Official sources

If you use the official sources, please pre-process the data following the instructions provided by [OpenTAD](https://github.com/sming256/OpenTAD).

| Dataset               | Task                          | Source                                                                                                                                                                                                                                                                                            |
|-----------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Charades-STA          | Video temporal grounding      | [Charades Official](https://prior.allenai.org/projects/charades),[Charades from OpenTAD](https://github.com/sming256/OpenTAD/blob/main/tools/prepare_data/charades/README.md), [Charades-STA annotations](https://opendatalab.com/OpenDataLab/Charades-STA/tree/main/w)                           |
| ActivityNet-Grounding | Video temporal grounding      | [ActivityNet Official](https://activity-net.org/download.html), [ActivityNet from OpenTAD](https://github.com/sming256/OpenTAD/tree/main/tools/prepare_data/activitynet), [ActivityNet Captions annotations](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions/tree/main/raw_data) |
| Video-MME             | Video question answering      | [Video-MME Official](https://huggingface.co/datasets/lmms-lab/Video-MME)                                                                                                                                                                                                                          |
| LongVideoBench        | Long-video question answering | [LongVideoBench Official](https://huggingface.co/datasets/longvideobench/LongVideoBench)                                                                                                                                                                                                          |

## Expected layout

The default configs expect:

```text
data/
|-- ActivityNet/
|   |-- captions/
|   |   `-- val_2.json
|   `-- rgb_videos_15fps_short256/
|       `-- <video_id>.mp4
|-- Charades/
|   |-- sta_annotation/
|   |   `-- charades_sta_test.txt
|   `-- rgb_videos_30fps_480/
|       `-- <video_id>.mp4
|-- VideoMME/
|   |-- test-00000-of-00001.parquet
|   |-- data/
|   |   `-- <video_id>.mp4
|   `-- subtitle/
`-- LongVideoBench/
    |-- lvb_val.json
    |-- videos/
    `-- subtitles/
```

The exact portable defaults live in [`configs/_base_/datasets`](../configs/_base_/datasets).

## Configure annotation and video paths

Before evaluation, open the preset for your dataset and edit:

- `eval_data_path`: annotation file (`.txt`, `.json`, or `.parquet`)
- `eval_video_folder`: directory containing the videos
- `subtitle_folder`: subtitle directory, required only by LongVideoBench

```python
# configs/_base_/datasets/charades_sta.py
eval_args = dict(
    dataset_name="Charades-STA",
    eval_data_path="/datasets/Charades/sta_annotation/charades_sta_test.txt",
    eval_video_folder="/datasets/Charades/rgb_videos_30fps_480",
    video_name_postfix=".mp4",
    prompt_task="temporal_grounding",
    is_shuffle=False,
)
```

Dataset paths belong only in `configs/_base_/datasets/`; method and model presets should remain machine-independent.

## Use `--data-root` instead

If you keep the released relative paths unchanged, `--data-root` rebases every configured path that starts with `data/`, including `eval_data_path`, `eval_video_folder`, and `subtitle_folder`:

```bash
python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets \
  --sample-amount 20 \
  --gpus 1
```

For this example, the loader resolves:

```text
/datasets/Charades/sta_annotation/charades_sta_test.txt
/datasets/Charades/rgb_videos_30fps_480/
```

`--data-root` intentionally leaves absolute or otherwise customized paths untouched. Use one approach per dataset: either edit its dataset preset or keep the `data/...` defaults and pass `--data-root`.

## Switch a comparison config to another dataset

Comparison-method examples use Charades-STA by default. To run one on ActivityNet, copy the config and change its
dataset base:

```python
_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen2_5_vl_7b.py",
    "../_base_/methods/tome_qwen2_5.py",
    "../_base_/datasets/activitynet_grounding.py",
]
```

Also give the copied config a distinct `training_args.output_dir`.

## Annotation cache

The loader accepts JSON, text, and parquet annotations. On first use, text and parquet inputs are normalized to a JSON
cache beside the annotation file. The annotation directory therefore needs write permission. Subsequent runs reuse that
cache.

Next: [evaluation and multi-GPU configuration](EVALUATION.md).
