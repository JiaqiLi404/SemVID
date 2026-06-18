import torch
import torch.nn as nn
from ..builder import LOSSES
from ..utils.iou_tools import compute_diou_torch, compute_giou_torch


@LOSSES.register_module()
class DIOULoss(nn.Module):
    def __init__(self):
        super(DIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
        weight: torch.Tensor=None
    ) -> torch.Tensor:
        diou=compute_diou_torch(target_bboxes, input_bboxes)
        loss = 1 - torch.diag(diou)
        diou=loss.clone()
        if weight is not None:
            loss = loss * weight

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss,diou


@LOSSES.register_module()
class GIOULoss(nn.Module):
    def __init__(self):
        super(GIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_giou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss
