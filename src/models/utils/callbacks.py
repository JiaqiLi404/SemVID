import torch
import logging
import os
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

def _bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 2)


class CUDAMemoryCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 10, reset_peak: bool = False):
        self.every_n_steps = every_n_steps
        self.reset_peak = reset_peak

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("[cuda-mem] reset peak stats")

    def _collect(self):
        if not torch.cuda.is_available():
            return None
        alloc_peak = torch.cuda.max_memory_allocated()
        reserv_peak = torch.cuda.max_memory_reserved()
        alloc = torch.cuda.memory_allocated()
        reserv = torch.cuda.memory_reserved()
        return {
            "cuda/alloc_peak_gb": _bytes_to_gb(alloc_peak),
            "cuda/reserv_peak_gb": _bytes_to_gb(reserv_peak),
            "cuda/alloc_gb": _bytes_to_gb(alloc),
            "cuda/reserv_gb": _bytes_to_gb(reserv),
        }

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            stats = self._collect()
            if stats is not None and getattr(state, "is_world_process_zero", True):
                logger.info(f"[cuda-mem][step={state.global_step}] {stats}")
                if self.reset_peak:
                    torch.cuda.reset_peak_memory_stats()
        return control

class LoggerMetricsCallback(TrainerCallback):
    def __init__(self, prefix: str = None):
        self.prefix = prefix

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        items = dict(logs)
        for k, v in list(items.items()):
            if hasattr(v, "item"):
                items[k] = v.item()

        # 格式化输出：key=value, 按 key 排序更稳定
        msg = ", ".join(f"{k}={items[k]}" for k in items.keys())
        if self.prefix:
            logger.info(f"[{self.prefix}] {msg}")
        else:
            logger.info(f"{msg}")