"""
src/training/hparam_tuning.py
──────────────────────────────
Automated hyperparameter search using Optuna.

Search space
────────────
  backbone         : efficientnet_b0 | mobilenet_v3_small
  lr_phase1        : log-uniform [1e-4, 5e-3]
  lr_phase2        : log-uniform [1e-6, 5e-4]
  focal_gamma      : float [1.0, 4.0]
  focal_alpha      : float [0.5, 0.9]
  dropout1         : float [0.2, 0.6]
  dropout2         : float [0.1, 0.4]
  hidden_dim       : categorical [128, 256, 512]
  weight_decay     : log-uniform [1e-5, 1e-3]
  batch_size       : categorical [32, 64, 128]
  augment_rotation : categorical [90, 180]

Objective: maximise val_f1 (with penalty if val_recall_faulty < 0.99).

Usage
─────
  python -m src.training.hparam_tuning \
      --config configs/config.yaml \
      --n_trials 50 \
      --n_epochs 15 \
      --study_name bottle_hparam_v1
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import optuna
import pytorch_lightning as pl
import yaml

from src.data.dataset import BottleDataModule
from src.training.lightning_module import BottleLightningModule

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    n_epochs: int = 15,
) -> float:
    """
    Single Optuna trial.

    Returns
    -------
    float — val_f1 score (higher is better).
            Returns 0.0 if the safety constraint (recall_faulty ≥ 0.99)
            is violated so Optuna prunes that direction.
    """
    cfg = copy.deepcopy(base_cfg)

    # ── Sample hyperparameters ────────────────────────────────────────
    backbone = trial.suggest_categorical(
        "backbone", ["efficientnet_b0", "mobilenet_v3_small"]
    )
    lr_p1 = trial.suggest_float("lr_phase1", 1e-4, 5e-3, log=True)
    lr_p2 = trial.suggest_float("lr_phase2", 1e-6, 5e-4, log=True)
    focal_gamma  = trial.suggest_float("focal_gamma", 1.0, 4.0)
    focal_alpha  = trial.suggest_float("focal_alpha", 0.5, 0.9)
    dropout1     = trial.suggest_float("dropout1", 0.2, 0.6)
    dropout2     = trial.suggest_float("dropout2", 0.1, 0.4)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
    rotation_deg = trial.suggest_categorical("rotation_degrees", [90, 180])

    # ── Patch config ──────────────────────────────────────────────────
    cfg["model"]["backbone"]                        = backbone
    cfg["model"]["head"]["dropout1"]               = dropout1
    cfg["model"]["head"]["dropout2"]               = dropout2
    cfg["model"]["head"]["hidden_dim"]             = hidden_dim
    cfg["training"]["phase1"]["lr"]                = lr_p1
    cfg["training"]["phase2"]["lr"]                = lr_p2
    cfg["training"]["loss"]["focal_gamma"]         = focal_gamma
    cfg["training"]["loss"]["focal_alpha"]         = focal_alpha
    cfg["training"]["optimizer"]["weight_decay"]   = weight_decay
    cfg["training"]["batch_size"]                  = batch_size
    cfg["training"]["epochs"]                      = n_epochs
    # Phase 1 gets 20% of budget, rest is phase 2
    cfg["training"]["phase1"]["epochs"]            = max(2, int(n_epochs * 0.20))
    cfg["training"]["phase2"]["epochs"]            = n_epochs - cfg["training"]["phase1"]["epochs"]
    cfg["augmentation"]["random_rotation"]["degrees"] = rotation_deg

    # Disable early stopping during search (fixed budget)
    cfg["training"]["early_stopping"]["enabled"]   = False

    # ── Data ─────────────────────────────────────────────────────────
    dm = BottleDataModule(cfg)
    dm.setup()

    # ── Module ───────────────────────────────────────────────────────
    module = BottleLightningModule(cfg, pos_weight=dm.pos_weight)

    # ── Trainer ───────────────────────────────────────────────────────
    callbacks = [
        optuna.integration.PyTorchLightningPruningCallback(
            trial, monitor="val_f1"
        )
    ]
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
        precision="16-mixed",
    )

    trainer.fit(module, datamodule=dm)

    # ── Retrieve metrics ──────────────────────────────────────────────
    metrics = trainer.callback_metrics
    val_f1     = float(metrics.get("val_f1",     0.0))
    val_recall = float(metrics.get("val_recall", 0.0))

    # Safety constraint: if recall on FAULTY class < 0.99, penalise
    min_recall = cfg["threshold"].get("min_recall_faulty", 0.99)
    if val_recall < min_recall:
        logger.debug(
            "Trial %d pruned: recall=%.4f < min_recall=%.4f",
            trial.number, val_recall, min_recall,
        )
        return 0.0

    logger.info(
        "Trial %d — F1=%.4f  Recall=%.4f  backbone=%s  lr_p1=%.2e  focal_γ=%.2f",
        trial.number, val_f1, val_recall, backbone, lr_p1, focal_gamma,
    )
    return val_f1


def run_study(
    config_path: str,
    n_trials: int = 50,
    n_epochs: int = 15,
    study_name: str = "bottle_hparam",
    storage: Optional[str] = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """
    Create and run an Optuna study.

    Parameters
    ----------
    config_path : path to config.yaml
    n_trials    : number of Optuna trials
    n_epochs    : epochs per trial (short budget; full training uses config epochs)
    study_name  : Optuna study name
    storage     : Optuna database URL (e.g. "sqlite:///hparam.db"); None = in-memory
    n_jobs      : parallel jobs (requires storage != None)

    Returns
    -------
    optuna.Study with all trial results
    """
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=base_cfg["project"]["seed"])

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, n_epochs),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("Best trial #%d — val_f1=%.4f", best.number, best.value)
    logger.info("Best hyperparameters:")
    for k, v in best.params.items():
        logger.info("  %-25s %s", k, v)
    logger.info("=" * 60)

    return study


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Hyperparameter search for bottle inspection model")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--n_epochs",   type=int, default=15)
    parser.add_argument("--study_name", default="bottle_hparam_v1")
    parser.add_argument("--storage",    default=None,
                        help="Optuna DB URL e.g. sqlite:///hparam.db")
    parser.add_argument("--n_jobs",     type=int, default=1)
    args = parser.parse_args()

    run_study(
        config_path=args.config,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        study_name=args.study_name,
        storage=args.storage,
        n_jobs=args.n_jobs,
    )
