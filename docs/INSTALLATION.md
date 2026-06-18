# Installation

## Requirements

- Linux or WSL2
- Python 3.11
- NVIDIA GPU with CUDA 12.x
- FFmpeg available on `PATH`

The paper experiments used three NVIDIA L40 GPUs with 48 GB memory each. One-GPU evaluation is supported but takes longer; Qwen3-VL-8B and long ActivityNet videos may require a 48 GB GPU or a lower input-token limit.

> [!IMPORTANT]
> Use **Transformers 4.x only**. The released model implementations are incompatible with Transformers 5.x. The verified and pinned version is `transformers==4.57.1`.

## Create the environment

```bash
git clone https://github.com/JiaqiLi404/SemVID.git
cd SemVID

conda create -n openvps python=3.11 -y
conda activate openvps

sudo apt-get update
sudo apt-get install -y ffmpeg

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Install PyTorch for a different CUDA runtime only when required by your system. Keep the package versions in `requirements.txt`, especially Transformers, unchanged unless you also validate the vendored model implementations.

## Verify the environment

```bash
python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("visible GPUs:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
assert transformers.__version__.startswith("4.57.")
PY
```

## Model access

The public configs use these Hugging Face IDs:

- `Qwen/Qwen3-VL-4B-Thinking`
- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

They download on first use. To use a local checkpoint, pass `--model-path /path/to/checkpoint` when evaluating.

## Troubleshooting

- **`flash-attn` build failure:** install PyTorch first, confirm the CUDA compiler matches the PyTorch build, then reinstall with `--no-build-isolation`.
- **Transformers import/model errors:** run `python -c "import transformers; print(transformers.__version__)"`; versions 5.0 and newer are unsupported.
- **CUDA out of memory:** use a smaller backbone, lower `eval_args.max_model_len`, or reduce `script_args.total_pixels_token_length` through `--cfg-options`.
- **No GPU visible:** verify `nvidia-smi`, the WSL NVIDIA driver, and `CUDA_VISIBLE_DEVICES` before launching evaluation.

Next: [prepare datasets](DATA.md), then [run evaluation](EVALUATION.md).
