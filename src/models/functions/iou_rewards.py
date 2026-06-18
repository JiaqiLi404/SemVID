import numpy as np

from .base import BaseReward
from ..builder import FUNCTIONS


@FUNCTIONS.register_module()
class IoUIntervalReward(BaseReward):
    def __init__(self, weight=1.0, eps=1e-8, **kwargs):
        super(IoUIntervalReward, self).__init__()
        self.weight = weight
        self.eps = eps

    def __call__(
            self,
            predictions,
            solution,
            **kwargs
    ):
        rewards = []
        for pred, sol in zip(predictions, solution):
            reward = 0.0
            gt_s, gt_e = sol
            gt_s = float(gt_s)
            gt_e = float(gt_e)

            if pred is not None and None not in pred:
                p_s, p_e = pred
                p_s = float(p_s)
                p_e = float(p_e)

                inter = max(0.0, min(p_e, gt_e) - max(p_s, gt_s))
                union = max(p_e, gt_e) - min(p_s, gt_s)

                # safe IoU
                if union > 0.0:
                    reward = self.weight * inter / (union + self.eps)  # eps 防极小数

            rewards.append(reward)

        return rewards


@FUNCTIONS.register_module()
class IoUWithEndpointAlignmentReward(BaseReward):
    def __init__(self, weight=1.0, eps=1e-8, **kwargs):
        super(IoUWithEndpointAlignmentReward, self).__init__()
        self.weight = weight
        self.eps = eps

    def __call__(
            self,
            predictions,
            solution,
            durations,
            **kwargs
    ):
        rewards = []
        for pred, sol, dura in zip(predictions, solution, durations):
            reward = 0.0
            gt_s, gt_e = sol
            gt_s = float(gt_s)
            gt_e = float(gt_e)

            if pred is not None and None not in pred:
                p_s, p_e = pred
                p_s = float(p_s)
                p_e = float(p_e)

                inter = max(0.0, min(p_e, gt_e) - max(p_s, gt_s))
                union = max(p_e, gt_e) - min(p_s, gt_s)

                # safe IoU
                if union > 0.0:
                    iou = inter / (union + self.eps)  # eps 防极小数

                gt_start_norm = 1.0 * gt_s / dura
                gt_end_norm = 1.0 * gt_e / dura
                pred_start_norm = 1.0 * p_s / dura
                pred_end_norm = 1.0 * p_e / dura
                reward = self.weight * iou * (1 - abs(gt_start_norm - pred_start_norm)) * (
                        1 - abs(gt_end_norm - pred_end_norm))

            rewards.append(reward)

        return rewards


@FUNCTIONS.register_module()
class TimestampReward(BaseReward):
    """
    Timestamp reward:
      r = start_weight * clamp01(1 - |p_s - gt_s|/dura)
        + end_weight   * clamp01(1 - |p_e - gt_e|/dura)

    Range: [0, start_weight + end_weight]
    """

    def __init__(self, start_weight=1.0, end_weight=1.0, eps=1e-8, **kwargs):
        super().__init__()
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.eps = float(eps)

    def __call__(self, predictions, solution, durations, **kwargs):
        rewards = []
        for pred, sol, dura in zip(predictions, solution, durations):
            gt_s, gt_e = sol
            gt_s = float(gt_s)
            gt_e = float(gt_e)

            if pred is None or (isinstance(pred, (list, tuple)) and None in pred):
                rewards.append(0.0)
                continue

            p_s, p_e = pred
            p_s = float(p_s)
            p_e = float(p_e)

            denom = (gt_e - gt_s) + self.eps
            start_sim = 1.0 - np.abs(p_s - gt_s) / denom
            end_sim = 1.0 - np.abs(p_e - gt_e) / denom

            # clamp to [0,1]
            start_sim = float(np.clip(start_sim, 0.0, 1.0))
            end_sim = float(np.clip(end_sim, 0.0, 1.0))

            reward = self.start_weight * start_sim + self.end_weight * end_sim
            rewards.append(float(reward))

        return rewards
