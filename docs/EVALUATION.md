# Evaluation

## Config structure

Each runnable config composes four small presets:

```python
_base_ = [
    "../_base_/runtime.py",
    "../_base_/models/qwen3_vl_4b.py",
    "../_base_/methods/semvid_qwen3.py",
    "../_base_/datasets/charades_sta.py",
]
```

- `runtime.py`: generation and GPU defaults
- `models/`: backbone and evaluator
- `methods/`: pruning implementation and released hyperparameters
- `datasets/`: annotation, video, subtitle, and prompt settings

Runnable configs are grouped by method directly under `configs/`; there is no separate synthetic `eval` config layer.

## Smoke test

Start with a deterministic subset:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets/openvps \
  --sample-amount 20 \
  --gpus 1
```

## Full evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets/openvps \
  --gpus 1
```

For ActivityNet or another backbone, select the corresponding config in [`configs/SemVID`](../configs/SemVID).

## Multi-GPU evaluation

Evaluation creates one Ray worker per GPU. The number passed to `--gpus` must not exceed the devices exposed by `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_8b_activitynet.py \
  --data-root /datasets/openvps \
  --gpus 3
```

The dataset is split across workers and predictions are merged for scoring. `eval_batch_size=1` is recommended for the released Qwen evaluator.

You may set the default in a config instead:

```python
eval_args = dict(eval_gpus=3)
```

The command-line flag takes precedence.

## Common overrides

```text
--data-root DIR          rebase every portable data/... path
--model-path ID_OR_DIR   use another Hugging Face ID or local checkpoint
--output-dir DIR         write this run to a different output directory
--gpus N                 launch N one-GPU workers
--retention-ratio R      override SemVID's semantic retention ratio
--sample-amount N        use a deterministic N-sample subset
```

Use dotted overrides for all other fields:

```bash
python tools/evaluate.py CONFIG \
  --data-root /datasets/openvps \
  --cfg-options \
    eval_args.max_model_len=32768 \
    eval_args.gpu_memory_utilization=0.80
```

For comparison methods, use their implementation-specific key:

```bash
# FastVID
--cfg-options script_args.model_hyper_parameters.fastvid_retention_ratio=0.20

# VisionZip, ToME, TokenSculpt, DART, VScan, or PruneVID
--cfg-options script_args.model_hyper_parameters.retention_ratio=0.25
```

## Available comparison configs

| Method | Config |
|---|---|
| Baseline | [`configs/Baseline`](../configs/Baseline) |
| FastVID | [`configs/FastVID`](../configs/FastVID) |
| VisionZip | [`configs/VisionZip`](../configs/VisionZip) |
| ToME | [`configs/ToME`](../configs/ToME) |
| TokenSculpt | [`configs/TokenSculpt`](../configs/TokenSculpt) |
| DART | [`configs/DART`](../configs/DART) |
| VScan | [`configs/VScan`](../configs/VScan) |
| PruneVID | [`configs/PruneVID`](../configs/PruneVID) |

These examples target Charades-STA. Follow [Data preparation](DATA.md#switch-a-comparison-config-to-another-dataset) to compose the same method with ActivityNet.

## VideoQA

```bash
# Qwen2.5-VL on Video-MME
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen2_5_vl_7b_videomme.py \
  --data-root /datasets/openvps --gpus 1

# LLaVA-OneVision on LongVideoBench
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/llava_onevision_7b_longvideobench.py \
  --data-root /datasets/openvps --gpus 1
```

## Outputs and resume behavior

Each run writes under:

```text
<training_args.output_dir>/eval_<dataset_name>/
|-- config.json
|-- pred_worker_*.jsonl
|-- predictions.jsonl
|-- predictions_with_score.json
|-- scores.json
`-- worker_*.log
```

VTG reports `mIoU` and `R1@{0.3, 0.5, 0.7}`. VideoQA reports overall and category accuracy. Interrupted runs resume from `pred_worker_*.jsonl`; use a new `--output-dir` when changing model, method, dataset, or retention ratio.

For the paper's graph diagnostics, continue with [ER/CS analysis](ANALYSIS.md).
