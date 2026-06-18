<div align="center">

# [ECCV 2026] Open Visual-Pruning Suite (SemVID)

### Keeping the Evidence Chain: Semantic Evidence Allocation for Training-Free Token Pruning in Video Temporal Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2603.05663-b31b1b.svg)](https://arxiv.org/abs/2603.05663)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](docs/INSTALLATION.md)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.1-FFD21E.svg)](docs/INSTALLATION.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[Paper](https://arxiv.org/pdf/2603.05663) · [Installation](docs/INSTALLATION.md) · [Data](docs/DATA.md) · [Evaluation](docs/EVALUATION.md)

</div>

Open Visual-Pruning Suite is the official implementation of **SemVID** and a unified evaluation codebase for training-free visual-token pruning on video temporal grounding and VideoQA. SemVID preserves a compact, temporally connected evidence chain by allocating tokens to query-relevant objects, motion transitions, and contextual anchors.

> **Headline result.** With Qwen3-VL-4B, SemVID retains up to **95.4% of the original mIoU using 12.5% of visual tokens**, while delivering up to **5.8× prefill speedup**.

## Why SemVID?

Video temporal grounding needs exact event boundaries and evidence that remains connected across time. Pruning objectives designed for VideoQA often preserve a few salient frames while dropping boundary cues or intermediate relay tokens. SemVID addresses this mismatch through:

- **Evidence Retention (ER):** keeps query-critical visual evidence, especially around temporal boundaries.
- **Connectivity Strength (CS):** preserves cross-frame attention paths for long-range evidence aggregation.
- **Training-free inference:** changes token selection without updating backbone weights.

## Quick start

### 1. Install

```bash
conda create -n openvps python=3.11 -y
conda activate openvps

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

> [!IMPORTANT]
> **Transformers must be below 5.0.** This release is verified with `transformers==4.57.1`; Transformers 5.x is incompatible with the released model implementations.

See [Installation](docs/INSTALLATION.md) for CUDA, FFmpeg, hardware, and troubleshooting notes.

### 2. Prepare data

Open the matching preset under [`configs/_base_/datasets`](configs/_base_/datasets) and set `eval_data_path` and `eval_video_folder` to your annotation and video locations. LongVideoBench also requires `subtitle_folder`:

```python
# configs/_base_/datasets/charades_sta.py
eval_args = dict(
    eval_data_path="/path/to/Charades/sta_annotation/charades_sta_test.txt",
    eval_video_folder="/path/to/Charades/rgb_videos_30fps_480",
    # Keep the remaining released settings unchanged.
)
```

Alternatively, keep the portable defaults and mirror this layout under one root:

```text
data/
|-- ActivityNet/
|   |-- captions/val_2.json
|   `-- rgb_videos_15fps_short256/
|-- Charades/
|   |-- sta_annotation/charades_sta_test.txt
|   `-- rgb_videos_30fps_480/
|-- VideoMME/
|   |-- test-00000-of-00001.parquet
|   `-- data/
`-- LongVideoBench/
    |-- lvb_val.json
    |-- videos/
    `-- subtitles/
```

Then pass the parent directory with `--data-root`; the launcher rebases annotation, video, and subtitle paths automatically. Official download links, exact layouts, and path rules are in [Data preparation](docs/DATA.md).

### 3. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /path/to/data \
  --gpus 1
```

Results are written to the config's `training_args.output_dir`. Multi-GPU launch, config overrides, dataset switching, and output formats are documented in [Evaluation](docs/EVALUATION.md).

## Included methods

| Method | Qwen3-VL | Qwen2.5-VL | LLaVA-OneVision | Public config |
|---|:---:|:---:|:---:|---|
| **SemVID** | ✓ | ✓ | ✓ | [`configs/SemVID`](configs/SemVID) |
| Sampling baselines | ✓ | - | - | [`configs/Baseline`](configs/Baseline) |
| FastVID | ✓ | ✓ | - | [`configs/FastVID`](configs/FastVID) |
| VisionZip | ✓ | ✓ | - | [`configs/VisionZip`](configs/VisionZip) |
| ToME | - | ✓ | - | [`configs/ToME`](configs/ToME) |
| TokenSculpt | - | ✓ | - | [`configs/TokenSculpt`](configs/TokenSculpt) |
| DART | - | ✓ | - | [`configs/DART`](configs/DART) |
| VScan | - | ✓ | - | [`configs/VScan`](configs/VScan) |
| PruneVID | - | ✓ | - | [`configs/PruneVID`](configs/PruneVID) |

The method presets preserve the released code settings. Dataset and model paths remain portable and can be changed without editing model source files.

## Evidence analysis

The paper's ER and CS implementations live in [`analysis/evidence_metrics.py`](analysis/evidence_metrics.py) and are called directly by the Qwen evaluator. A full-token pass establishes the reference landing distribution; a pruned pass then produces per-sample `attn_metrics.json` files.

```bash
python analysis/evidence_metrics.py analysis_outputs/qwen3_charades/samples
```

See [ER/CS analysis](docs/ANALYSIS.md) for the complete two-pass workflow and metric definitions.

## Repository layout

```text
OpenVPS/
|-- analysis/       # Evidence Retention and Connectivity Strength
|-- configs/        # Method-oriented, composable experiment configs
|-- docs/           # Installation, data, evaluation, and analysis guides
|-- src/            # Datasets, evaluators, and model implementations
|-- tools/
|   `-- evaluate.py # Evaluation entry point
|-- CITATION.cff
|-- LICENSE
`-- requirements.txt
```

## Documentation

| Guide | Covers |
|---|---|
| [Installation](docs/INSTALLATION.md) | Environment, CUDA stack, Transformers constraint, verification |
| [Data preparation](docs/DATA.md) | Official datasets, directory structure, path configuration |
| [Evaluation](docs/EVALUATION.md) | Config composition, single/multi-GPU runs, overrides, outputs |
| [ER/CS analysis](docs/ANALYSIS.md) | Full-token reference pass, pruned pass, metric aggregation |

## Citation

```bibtex
@article{li2026keeping,
  title     = {Keeping the Evidence Chain: Semantic Evidence Allocation for Training-Free Token Pruning in Video Temporal Grounding},
  author    = {Li, Jiaqi and Zheng, Shuntian and Shen, Yixian and Huang, Jia-Hong and Lu, Xiaoman and Ni, Minzhe and Guan, Yu},
  journal   = {arXiv preprint arXiv:2603.05663},
  year      = {2026}
}
```

## Acknowledgements

This project builds on [Hugging Face Transformers](https://github.com/huggingface/transformers), [Qwen-VL](https://github.com/QwenLM/Qwen-VL), [LLaVA-OneVision](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2), [OpenTAD](https://github.com/sming256/OpenTAD), and the visual-token pruning methods included in the suite.

## License

Code is released under the [Apache License 2.0](LICENSE). Model weights and datasets remain subject to their original licenses and access terms; see [NOTICE](NOTICE).
