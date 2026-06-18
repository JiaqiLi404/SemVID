# Evidence Retention and Connectivity Strength

The original ER/CS implementation has been extracted from the Qwen evaluator into [`analysis/evidence_metrics.py`](../analysis/evidence_metrics.py). The evaluator imports these functions directly, so online metric generation and offline aggregation share one implementation.

## What is computed

- `compute_landing_distribution`: propagates query evidence backward through sparse visual self-attention (Eq. 14).
- `compute_connectivity_strength`: sums retained adjacent-frame transition mass per layer (Eqs. 12, 13, and 15).
- `compute_evidence_cross_entropy`: computes the full/pruned evidence cross-entropy term used by ER (Eq. 11).
- `evidence_retention_score`: applies the paper's reporting transform, `exp(-cross_entropy) × 10,000`.

Evidence analysis is implemented for the instrumented Qwen3 SemVID model.

## Two-pass workflow

ER compares a pruned landing distribution with the matching full-token distribution. Both passes must use the same model, dataset, preprocessing, sample subset, and analysis root. Use different evaluation output directories so resume logic does not skip the second pass.

### 1. Generate the full-token reference

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets/openvps \
  --sample-amount 100 \
  --gpus 1 \
  --retention-ratio 1.0 \
  --output-dir outputs/analysis/qwen3_full \
  --analyze-evidence \
  --analysis-root analysis_outputs/qwen3_charades
```

This creates the full-token landing maps under:

```text
analysis_outputs/qwen3_charades/attention_map/
```

### 2. Run the pruned method

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  configs/SemVID/qwen3_vl_4b_charades.py \
  --data-root /datasets/openvps \
  --sample-amount 100 \
  --gpus 1 \
  --retention-ratio 0.125 \
  --output-dir outputs/analysis/qwen3_semvid_r0125 \
  --analyze-evidence \
  --analysis-root analysis_outputs/qwen3_charades
```

Per-sample metrics are written to:

```text
analysis_outputs/qwen3_charades/samples/<video_id>_<sample_hash>/attn_metrics.json
```

Each record contains the original sample ID, the raw ER cross-entropy, mean/per-layer CS, and per-transition connectivity values. The hash keeps full-token references separate when one video has multiple queries.

### 3. Aggregate the paper metrics

```bash
python analysis/evidence_metrics.py \
  analysis_outputs/qwen3_charades/samples \
  --output analysis_outputs/qwen3_charades/summary.json
```

Example output:

```json
{
  "metric_files": 100,
  "er_samples": 100,
  "cs_samples": 100,
  "ER": 5.6,
  "CS": 33.5
}
```

## Avoid stale references

Use a distinct `--analysis-root` whenever the backbone, dataset, preprocessing, or sample subset changes. If a run was interrupted while creating the full-token reference, remove that analysis directory and rerun the reference pass before computing pruned ER.
