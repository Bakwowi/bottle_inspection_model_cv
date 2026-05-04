"""
scripts/train.py
─────────────────
Main training entry point.

Usage
─────
  # Standard run
  python scripts/train.py --config configs/config.yaml

  # Resume from checkpoint
  python scripts/train.py --config configs/config.yaml \
      --resume checkpoints/last.ckpt

  # Override single config values from CLI
  python scripts/train.py \
      --config configs/config.yaml \
      --override training.batch_size=32 model.backbone=mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger

from src.data.dataset import BottleDataModule
from src.training.lightning_module import BottleLightningModule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation overrides, e.g. 'training.batch_size=64'.
    Supports nested keys separated by '.'.
    """
    for override in overrides:
        key_path, _, value_str = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # Attempt type inference
        leaf_key = keys[-1]
        existing = node.get(leaf_key)
        if isinstance(existing, bool):
            node[leaf_key] = value_str.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            node[leaf_key] = int(value_str)
        elif isinstance(existing, float):
            node[leaf_key] = float(value_str)
        else:
            node[leaf_key] = value_str
        logger.info("Config override: %s = %s", key_path, node[leaf_key])
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Logger factory
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(cfg: dict):
    log_cfg  = cfg.get("logging", {})
    log_type = log_cfg.get("logger", "csv").lower()
    proj     = log_cfg.get("project", "bottle_inspection")
    name     = cfg["project"].get("name", "run")

    if log_type == "wandb":
        try:
            import wandb  # noqa: F401
            return WandbLogger(project=proj, name=name)
        except ImportError:
            logger.warning("wandb not installed, falling back to TensorBoard.")
            log_type = "tensorboard"

    if log_type == "tensorboard":
        return TensorBoardLogger(save_dir="logs/", name=name)

    # Default: CSV (always works, no external dependency)
    return CSVLogger(save_dir="logs/", name=name)


# ─────────────────────────────────────────────────────────────────────────────
# Callback factory
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(cfg: dict) -> list:
    train_cfg = cfg["training"]
    ckpt_cfg  = train_cfg["checkpoint"]
    es_cfg    = train_cfg["early_stopping"]

    callbacks = []

    # Model checkpoint — save top-k by val_f1
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_cfg.get("dirpath", "checkpoints/"),
            filename="epoch{epoch:03d}-val_f1{val_f1:.4f}",
            monitor=ckpt_cfg.get("monitor", "val_f1"),
            mode=ckpt_cfg.get("mode", "max"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True,
        )
    )

    # Early stopping
    if es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_f1"),
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.0005),
                mode=es_cfg.get("mode", "max"),
                verbose=True,
            )
        )

    # LR monitoring (shows current LR in progress bar)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Rich progress bar (pretty output; falls back gracefully if not installed)
    try:
        import rich  # noqa: F401
        callbacks.append(RichProgressBar())
    except ImportError:
        pass

    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, resume_from: str | None = None):
    """
    Full training pipeline.

    Parameters
    ----------
    cfg         : full config dictionary
    resume_from : optional path to a .ckpt to resume from
    """
    # ── Reproducibility ──────────────────────────────────────────────
    seed = cfg["project"].get("seed", 42)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")   # speeds up matmul on Ampere+

    # ── DataModule ───────────────────────────────────────────────────
    logger.info("Setting up data module...")
    dm = BottleDataModule(cfg)
    dm.setup()

    # ── Lightning module ─────────────────────────────────────────────
    logger.info("Building model...")
    module = BottleLightningModule(cfg, pos_weight=dm.pos_weight)

    n_total     = module.model.total_param_count()
    n_trainable = module.model.trainable_param_count()
    logger.info("Parameters — total: %d | trainable (phase 1): %d", n_total, n_trainable)

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if cfg["training"].get("amp", True) else "32",
        accumulate_grad_batches=cfg["training"].get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg["training"].get("gradient_clip_val", 1.0),
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg),
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
        deterministic=False,   # True is slower; off for speed
        benchmark=True,        # cudnn auto-tune (faster for fixed input size)
    )

    # ── Fit ───────────────────────────────────────────────────────────
    logger.info("Starting training (epochs=%d)...", cfg["training"]["epochs"])
    trainer.fit(module, datamodule=dm, ckpt_path=resume_from)

    # ── Summary ───────────────────────────────────────────────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    logger.info("Training complete.")
    logger.info("Best checkpoint : %s", best_ckpt)
    logger.info("Best val_f1     : %.4f", best_score)

    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/train.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Train the bottle inspection classifier")
    parser.add_argument("--config",   default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--resume",   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key.subkey=value (space-separated)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    best_ckpt = train(cfg, resume_from=args.resume)
    print(f"\nBest checkpoint saved at: {best_ckpt}")

import json
txt = "hello world"
