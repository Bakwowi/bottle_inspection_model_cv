"""
src/models/classifier.py
─────────────────────────
Binary bottle-inspection classifier built on a pretrained CNN backbone
with a lightweight two-layer classification head.

Supported backbones
───────────────────
  efficientnet_b0      ← default; best accuracy/speed tradeoff
  mobilenet_v3_small   ← faster, marginally lower ceiling
  resnet50             ← baseline / ablation

Head architecture
─────────────────
  backbone → GlobalAveragePooling (implicit in timm)
           → Dropout(p1)
           → Linear(feat_dim → hidden_dim) + BatchNorm1d + ReLU
           → Dropout(p2)
           → Linear(hidden_dim → 1)
           → Sigmoid  (applied during inference; BCEWithLogitsLoss during training)
"""

from __future__ import annotations

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Feature dimensions of the GAP output for each backbone
_FEATURE_DIMS = {
    "efficientnet_b0":    1280,
    "mobilenet_v3_small":  576,
    "resnet50":           2048,
}


class BottleClassifier(nn.Module):
    """
    Parameters
    ----------
    backbone    : timm model name (see _FEATURE_DIMS above)
    pretrained  : load ImageNet weights
    hidden_dim  : neurons in the intermediate FC layer
    dropout1    : dropout after backbone GAP
    dropout2    : dropout after first FC
    """

    def __init__(
        self,
        backbone: str   = "efficientnet_b0",
        pretrained: bool = True,
        hidden_dim: int  = 256,
        dropout1: float  = 0.4,
        dropout2: float  = 0.2,
    ):
        super().__init__()

        if backbone not in _FEATURE_DIMS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(_FEATURE_DIMS.keys())}"
            )

        # ── Backbone (feature extractor) ─────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,        # removes the original head → outputs features
            global_pool="avg",    # Global Average Pooling
        )
        feat_dim = _FEATURE_DIMS[backbone]

        # ── Classification head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, 1),
            # No Sigmoid here — BCEWithLogitsLoss is numerically stable
            # Sigmoid is applied externally during inference
        )

        # Weight initialisation for the head
        self._init_head()

        logger.info(
            "BottleClassifier | backbone=%s | feat_dim=%d | head_hidden=%d",
            backbone, feat_dim, hidden_dim,
        )

    # ------------------------------------------------------------------
    def _init_head(self):
        """Kaiming uniform for linear layers, constant for BN."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : float32 tensor  [B, 3, H, W]

        Returns
        -------
        logits : float32 tensor  [B, 1]   (raw, pre-sigmoid)
        """
        features = self.backbone(x)   # [B, feat_dim]
        logits   = self.head(features) # [B, 1]
        return logits

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference: returns P(FAULTY) in [0, 1].
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze all backbone parameters (phase-1 warmup)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self, last_n_blocks: Optional[int] = None):
        """
        Unfreeze backbone parameters.

        Parameters
        ----------
        last_n_blocks : if None, unfreeze everything;
                        if int, unfreeze only the last N child modules
                        (useful for EfficientNet which has numbered blocks).
        """
        if last_n_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen.")
        else:
            children = list(self.backbone.children())
            n = len(children)
            # Unfreeze last `last_n_blocks` children
            for i, child in enumerate(children):
                if i >= n - last_n_blocks:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False
            logger.info("Last %d backbone blocks unfrozen.", last_n_blocks)

    # ------------------------------------------------------------------
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
