"""
src/models/losses.py
─────────────────────
Focal Loss for binary classification.

Focal Loss (Lin et al. 2017) addresses class imbalance by down-weighting
easy examples so the model focuses on hard-to-classify borderline samples —
exactly the profile of this task (most bottles are good; rare defects are
the hard, important cases).

  FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

  α : class balance weight  (higher → more weight on positive/faulty class)
  γ : focusing parameter    (higher → harder examples weighted more)
      γ=0 reduces to standard BCE; γ=2 is the original paper's default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss operating on raw logits.

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard BCE, 2 = original paper default.
    alpha : float
        Weight for the positive (FAULTY) class. Set below 0.5 to penalise
        false negatives more; typical range 0.5–0.85 for imbalanced data.
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : raw model output  [B] or [B, 1]
        targets : binary float targets {0., 1.}  [B] or [B, 1]
        """
        logits  = logits.view(-1)
        targets = targets.view(-1)

        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)

        # α_t: per-sample alpha weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal weight
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """
    Factory that returns the configured loss function.

    Parameters
    ----------
    cfg        : training.loss section of config.yaml
    pos_weight : computed from dataset statistics (used for weighted BCE)
    """
    name = cfg.get("name", "focal").lower()

    if name == "focal":
        return BinaryFocalLoss(
            gamma=cfg.get("focal_gamma", 2.0),
            alpha=cfg.get("focal_alpha", 0.75),
        )

    elif name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight must be provided for weighted_bce loss.")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    else:
        raise ValueError(f"Unknown loss function: '{name}'")
