# Data preparation

Open Visual-Pruning Suite does not redistribute videos, annotations, or subtitles. Download each benchmark from its official source and follow its license and access terms.

## Official sources

| Dataset | Task | Source |
|---|---|---|
| Charades-STA | Video temporal grounding | [Charades](https://prior.allenai.org/projects/charades), [Charades-STA annotations](https://github.com/jiyanggao/TALL) |
| ActivityNet-Grounding | Video temporal grounding | [ActivityNet](https://activity-net.org/download.html), ActivityNet Captions/Grounding annotations |
| Video-MME | Video question answering | [Video-MME on Hugging Face](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| LongVideoBench | Long-video question answering | [LongVideoBench on Hugging Face](https://huggingface.co/datasets/longvideobench/LongVideoBench) |

LongVideoBench is gated and requires access approval.

## Expected layout

The default configs expect:

```text
data/
|-- charades_sta/
|   |-- charades_sta_test.txt
|   `-- videos/
|       `-- <video_id>.mp4
|-- activitynet_grounding/
|   |-- val_2.json
|   `-- videos/
|       `-- <video_id>.mp4
|-- videomme/
|   |-- test-00000-of-00001.parquet
|   `-- data/
|       `-- <video_id>.mp4
`-- longvideobench/
    |-- lvb_val.json
    |-- videos/
    `-- subtitles/
```

The exact portable defaults live in [`configs/_base_/datasets`](../configs/_base_/datasets).

## Keep data outside the repository

Use `--data-root` to rebase every default `data/...` path:

```bash
python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets/openvps \
  --sample-amount 20 \
  --gpus 1
```

For this example, the loader resolves:

```text
/datasets/openvps/charades_sta/charades_sta_test.txt
/datasets/openvps/charades_sta/videos/
```

You can instead edit the paths in the relevant dataset preset. Do not put machine-specific absolute paths in method or model files.

## Switch a comparison config to another dataset

Comparison-method examples use Charades-STA by default. To run one on ActivityNet, copy the config and change its dataset base:

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

The loader accepts JSON, text, and parquet annotations. On first use, text and parquet inputs are normalized to a JSON cache beside the annotation file. The annotation directory therefore needs write permission. Subsequent runs reuse that cache.

Next: [evaluation and multi-GPU configuration](EVALUATION.md).
