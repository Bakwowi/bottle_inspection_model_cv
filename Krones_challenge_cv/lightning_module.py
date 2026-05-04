"""
src/training/lightning_module.py
──────────────────────────────────
PyTorch-Lightning LightningModule that wraps the BottleClassifier.

Two-phase training strategy
────────────────────────────
Phase 1 (epochs 0 … phase1.epochs):
  - Backbone frozen
  - Only the classification head is trained
  - Higher learning rate (warms up head weights)

Phase 2 (epochs phase1.epochs … total_epochs):
  - Last N backbone blocks unfrozen
  - Full model fine-tuned at lower LR with cosine annealing

Scheduler
─────────
Cosine annealing with linear warmup. Implemented as a custom lambda
scheduler to avoid the complexity of chaining schedulers.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import (
    AUROC,
    F1Score,
    AveragePrecision,
    Precision,
    Recall,
    Accuracy,
)

from src.models.classifier import BottleClassifier
from src.models.losses import build_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup_fn(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 1e-7,
) -> float:
    """Lambda for torch.optim.lr_scheduler.LambdaLR."""
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine)


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────

class BottleLightningModule(pl.LightningModule):
    """
    Parameters
    ----------
    cfg        : full config dict (from config.yaml)
    pos_weight : optional tensor for weighted BCE; computed by DataModule
    """

    def __init__(self, cfg: dict, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.cfg = cfg
        self.train_cfg  = cfg["training"]
        self.model_cfg  = cfg["model"]
        self.thresh_cfg = cfg["threshold"]

        # ── Model ────────────────────────────────────────────────────
        head = self.model_cfg.get("head", {})
        self.model = BottleClassifier(
            backbone   = self.model_cfg["backbone"],
            pretrained = self.model_cfg["pretrained"],
            hidden_dim = head.get("hidden_dim", 256),
            dropout1   = head.get("dropout1", 0.4),
            dropout2   = head.get("dropout2", 0.2),
        )

        # Start with backbone frozen (phase 1)
        self.model.freeze_backbone()
        self._phase = 1

        # ── Loss ─────────────────────────────────────────────────────
        # Respect "auto" pos_weight: use the one computed from data
        if (
            self.train_cfg["loss"].get("name") == "weighted_bce"
            and self.train_cfg.get("pos_weight") == "auto"
            and pos_weight is not None
        ):
            self.criterion = build_loss(self.train_cfg["loss"], pos_weight)
        else:
            self.criterion = build_loss(self.train_cfg["loss"])

        # ── Metrics ───────────────────────────────────────────────────
        # torchmetrics v1 API — task="binary"
        metric_kwargs = dict(task="binary")

        for split in ("train", "val", "test"):
            setattr(self, f"{split}_f1",        F1Score(**metric_kwargs))
            setattr(self, f"{split}_precision",  Precision(**metric_kwargs))
            setattr(self, f"{split}_recall",     Recall(**metric_kwargs))
            setattr(self, f"{split}_accuracy",   Accuracy(**metric_kwargs))
            setattr(self, f"{split}_auroc",      AUROC(**metric_kwargs))
            setattr(self, f"{split}_pr_auc",     AveragePrecision(**metric_kwargs))

        # Default decision threshold (will be calibrated after training)
        self.register_buffer(
            "threshold",
            torch.tensor(self.thresh_cfg.get("fixed_value", 0.5)),
        )

        logger.info(
            "LightningModule ready | backbone=%s | loss=%s | phase=1",
            self.model_cfg["backbone"],
            self.train_cfg["loss"]["name"],
        )

    # ------------------------------------------------------------------
    # Phase switching
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        p1_epochs = self.train_cfg["phase1"]["epochs"]
        if self.current_epoch == p1_epochs and self._phase == 1:
            self._switch_to_phase2()

    def _switch_to_phase2(self):
        unfreeze = self.train_cfg["phase2"].get("unfreeze_layers", "last3blocks")
        if unfreeze == "all":
            self.model.unfreeze_backbone(last_n_blocks=None)
        else:
            # Parse "last3blocks" → 3
            n = int("".join(filter(str.isdigit, unfreeze))) if any(
                c.isdigit() for c in unfreeze
            ) else 3
            self.model.unfreeze_backbone(last_n_blocks=n)
        self._phase = 2
        logger.info(
            "Epoch %d: switching to phase 2 — backbone unfrozen (%s). "
            "Trainable params: %d",
            self.current_epoch,
            unfreeze,
            self.model.trainable_param_count(),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(
        self, batch: tuple, split: str
    ) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images).squeeze(1)   # [B]
        loss   = self.criterion(logits, labels)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= self.threshold).long()
        labels_int = labels.long()

        # Update torchmetrics
        getattr(self, f"{split}_f1")(preds, labels_int)
        getattr(self, f"{split}_precision")(preds, labels_int)
        getattr(self, f"{split}_recall")(preds, labels_int)
        getattr(self, f"{split}_accuracy")(preds, labels_int)
        getattr(self, f"{split}_auroc")(probs, labels_int)
        getattr(self, f"{split}_pr_auc")(probs, labels_int)

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Train / Val / Test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, "train")
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return out["loss"]

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
        self.log("val_loss", out["loss"], prog_bar=True, sync_dist=True)
        return out["loss"]

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch, "test")
        self.log("test_loss", out["loss"], sync_dist=True)
        return out["loss"]

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    # ------------------------------------------------------------------
    def _log_epoch_metrics(self, split: str):
        f1   = getattr(self, f"{split}_f1").compute()
        prec = getattr(self, f"{split}_precision").compute()
        rec  = getattr(self, f"{split}_recall").compute()
        acc  = getattr(self, f"{split}_accuracy").compute()
        auc  = getattr(self, f"{split}_auroc").compute()
        pr   = getattr(self, f"{split}_pr_auc").compute()

        metrics = {
            f"{split}_f1":        f1,
            f"{split}_precision": prec,
            f"{split}_recall":    rec,
            f"{split}_accuracy":  acc,
            f"{split}_auroc":     auc,
            f"{split}_pr_auc":    pr,
        }
        self.log_dict(metrics, prog_bar=(split == "val"), sync_dist=True)

        # Reset for next epoch
        for name in ("f1", "precision", "recall", "accuracy", "auroc", "pr_auc"):
            getattr(self, f"{split}_{name}").reset()

        if split == "val":
            logger.info(
                "Epoch %d [val] F1=%.4f  Prec=%.4f  Rec=%.4f  ACC=%.4f  AUC=%.4f  PR-AUC=%.4f",
                self.current_epoch, f1, prec, rec, acc, auc, pr,
            )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg  = self.train_cfg["optimizer"]
        sch_cfg  = self.train_cfg["scheduler"]
        p1_cfg   = self.train_cfg["phase1"]
        p2_cfg   = self.train_cfg["phase2"]

        # Phase-aware learning rate: phase 1 LR at epoch 0
        initial_lr = p1_cfg["lr"]

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=initial_lr,
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )

        # Estimate steps for scheduler
        # Trainer.estimated_stepping_batches is available after fit() starts
        # We compute it conservatively here
        total_epochs  = self.train_cfg["epochs"]
        warmup_epochs = sch_cfg.get("warmup_epochs", 3)

        # Use a per-epoch step count of 1 for simplicity;
        # LambdaLR will be called once per epoch via epoch-level scheduling
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self._lr_lambda(
                epoch, total_epochs, warmup_epochs, p1_cfg, p2_cfg,
                sch_cfg.get("min_lr", 1e-7),
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def _lr_lambda(
        epoch: int,
        total_epochs: int,
        warmup_epochs: int,
        p1_cfg: dict,
        p2_cfg: dict,
        min_lr: float,
    ) -> float:
        """
        Piecewise LR schedule:
          0 … warmup_epochs         : linear warmup to phase1.lr
          warmup_epochs … p1.epochs : constant phase1.lr
          p1.epochs … total_epochs  : cosine decay from phase2.lr to min_lr
        """
        p1_lr   = p1_cfg["lr"]
        p2_lr   = p2_cfg["lr"]
        p1_end  = p1_cfg["epochs"]

        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / max(1, warmup_epochs)

        if epoch < p1_end:
            return 1.0  # flat at phase1.lr

        # Phase 2: cosine from p2_lr down to min_lr
        p2_start  = p1_end
        p2_length = total_epochs - p2_start
        progress  = (epoch - p2_start) / max(1, p2_length)
        cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Rescale relative to phase1 base LR
        target_lr = min_lr + (p2_lr - min_lr) * cosine
        return target_lr / p1_lr  # LambdaLR multiplies by base LR (=p1_lr)
