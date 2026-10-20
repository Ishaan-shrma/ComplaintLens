"""
src/models/focal_loss.py
========================
Focal Loss with Label Smoothing for class-imbalanced and hard-example sentiment learning.

Formula:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
where p_t is the model's estimated probability for the smoothed target ground truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        num_classes = logits.size(-1)
        
        # Apply label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, fill_value=self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weighting: (1 - p)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)

        # Cross entropy loss with focal term
        loss = -focal_weight * smooth_targets * log_probs
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
