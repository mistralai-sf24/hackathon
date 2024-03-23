from typing import Optional

import torch
from torch.nn import functional as F


def compute_loss(logits: torch.Tensor, target: torch.Tensor, reduction: str):
    assert reduction in ["mean", "none"]
    mb_loss = F.cross_entropy(logits, target, reduction=reduction)
    return mb_loss


def compute_loss_with_mask(
    logits: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor]
):
    if target_mask is not None:
        mb_loss = compute_loss(logits, target, reduction="none")
        mb_loss = torch.sum(mb_loss * target_mask) / torch.sum(target_mask)
    else:
        mb_loss = compute_loss(logits, target, reduction="mean")
    return mb_loss
