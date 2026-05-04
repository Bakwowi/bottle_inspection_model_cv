"""
src/evaluation/evaluator.py
────────────────────────────
Full evaluation pipeline for the trained bottle inspection model.

Steps
─────
1. Load best checkpoint
2. Run inference on the test set (full pass with no augmentation)
3. Collect logits / probabilities / ground-truth labels
4. Calibrate the decision threshold via F1-maximisation
   (optionally with a minimum recall constraint on the FAULTY class)
5. Compute and print all metrics
6. Export predictions CSV
7. Save precision-recall and ROC curves

This module is the single source of truth for the model's reported performance.
It is deliberately separated from the Lightning trainer so it can be run
independently on any checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BottleDataModule, build_transforms
from src.models.classifier import BottleClassifier
from src.training.lightning_module import BottleLightningModule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model over the dataloader and collect probabilities + labels.

    Returns
    -------
    probs  : np.ndarray [N]  — P(FAULTY) in [0, 1]
    labels : np.ndarray [N]  — ground-truth {0, 1}
    """
    model.eval()
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    for images, targets in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(targets.numpy().astype(int).tolist())

    return np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = "f1_maximization",
    min_recall_faulty: float = 0.990,
    fixed_value: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal decision threshold.

    method
    ──────
    f1_maximization         : argmax F1 over candidate thresholds,
                              subject to recall_faulty ≥ min_recall_faulty
    precision_recall_tradeoff : same search but prefers higher precision
    fixed                   : use fixed_value as-is

    Returns
    -------
    threshold : float
    metrics   : dict of metrics at the chosen threshold
    """
    if method == "fixed":
        preds = (probs >= fixed_value).astype(int)
        return fixed_value, _metrics_at_threshold(labels, preds, probs)

    # Candidate thresholds from precision-recall curve
    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs)
    # thresholds has len N-1; precision/recall have len N — align
    thresholds = np.append(thresholds, 1.0)

    best_threshold = fixed_value
    best_f1        = 0.0
    best_metrics   = {}

    for tau, prec, rec in zip(thresholds, precision_arr, recall_arr):
        if rec < min_recall_faulty:
            continue   # safety constraint: must not miss too many faulty bottles
        preds  = (probs >= tau).astype(int)
        f1_val = f1_score(labels, preds, zero_division=0)

        if method == "precision_recall_tradeoff":
            score = 0.5 * f1_val + 0.5 * prec   # slightly prefer precision
        else:
            score = f1_val

        if score > best_f1:
            best_f1        = score
            best_threshold = float(tau)
            best_metrics   = _metrics_at_threshold(labels, preds, probs)

    if not best_metrics:
        logger.warning(
            "No threshold satisfies min_recall_faulty=%.3f. "
            "Relaxing to best available recall.",
            min_recall_faulty,
        )
        # Fall back: pick threshold that maximises recall, then F1
        for tau, prec, rec in zip(thresholds, precision_arr, recall_arr):
            preds  = (probs >= tau).astype(int)
            f1_val = f1_score(labels, preds, zero_division=0)
            if f1_val > best_f1:
                best_f1        = f1_val
                best_threshold = float(tau)
                best_metrics   = _metrics_at_threshold(labels, preds, probs)

    logger.info(
        "Calibrated threshold=%.4f | F1=%.4f | Recall(FAULTY)=%.4f | Prec(FAULTY)=%.4f",
        best_threshold,
        best_metrics.get("f1", 0),
        best_metrics.get("recall_faulty", 0),
        best_metrics.get("precision_faulty", 0),
    )
    return best_threshold, best_metrics


def _metrics_at_threshold(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Compute all scalar metrics given predictions."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision_f = tp / max(tp + fp, 1)
    recall_f    = tp / max(tp + fn, 1)
    precision_g = tn / max(tn + fn, 1)
    recall_g    = tn / max(tn + fp, 1)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_faulty = 2 * precision_f * recall_f / max(precision_f + recall_f, 1e-9)

    try:
        roc_auc = roc_auc_score(labels, probs)
        pr_auc  = average_precision_score(labels, probs)
    except Exception:
        roc_auc = pr_auc = 0.0

    accuracy = (tp + tn) / max(len(labels), 1)

    return {
        "f1":               f1_macro,
        "f1_faulty":        f1_faulty,
        "precision_faulty": precision_f,
        "recall_faulty":    recall_f,
        "precision_good":   precision_g,
        "recall_good":      recall_g,
        "accuracy":         accuracy,
        "roc_auc":          roc_auc,
        "pr_auc":           pr_auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Orchestrates full evaluation of a trained model checkpoint.

    Parameters
    ----------
    checkpoint_path : path to .ckpt file saved by Lightning
    config_path     : path to config.yaml
    output_dir      : where to save reports, plots, predictions CSV
    device          : 'cuda', 'cpu', or 'auto'
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        output_dir: str = "outputs/evaluation",
        device: str = "auto",
    ):
        self.ckpt_path  = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Evaluator | device=%s | ckpt=%s", self.device, self.ckpt_path)

    # ------------------------------------------------------------------
    def load_model(self) -> BottleClassifier:
        """Load model weights from Lightning checkpoint."""
        module = BottleLightningModule.load_from_checkpoint(
            self.ckpt_path,
            cfg=self.cfg,
            map_location=self.device,
        )
        model = module.model.to(self.device)
        model.eval()
        logger.info(
            "Model loaded — total params: %d | trainable: %d",
            model.total_param_count(),
            model.trainable_param_count(),
        )
        return model

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        """
        Full evaluation pipeline.

        Returns
        -------
        dict of final metrics
        """
        # ── Data ─────────────────────────────────────────────────────
        dm = BottleDataModule(self.cfg)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()

        # ── Model ────────────────────────────────────────────────────
        model = self.load_model()

        # ── Collect predictions ───────────────────────────────────────
        logger.info("Running inference on test set (%d samples)...", len(dm.test_ds))
        probs, labels = collect_predictions(
            model, test_loader, self.device,
            use_amp=self.cfg["training"].get("amp", True),
        )

        # ── Threshold calibration (on val set) ───────────────────────
        # Best practice: calibrate on val, report on test
        logger.info("Calibrating decision threshold on validation set...")
        val_loader = dm.val_dataloader()
        val_probs, val_labels = collect_predictions(
            model, val_loader, self.device,
            use_amp=self.cfg["training"].get("amp", True),
        )

        thr_cfg = self.cfg["threshold"]
        threshold, val_metrics = calibrate_threshold(
            val_probs, val_labels,
            method=thr_cfg.get("method", "f1_maximization"),
            min_recall_faulty=thr_cfg.get("min_recall_faulty", 0.990),
            fixed_value=thr_cfg.get("fixed_value", 0.5),
        )

        # ── Final test metrics ────────────────────────────────────────
        test_preds   = (probs >= threshold).astype(int)
        test_metrics = _metrics_at_threshold(labels, test_preds, probs)

        # ── Report ───────────────────────────────────────────────────
        self._print_report(test_metrics, threshold, labels, test_preds)
        self._save_classification_report(labels, test_preds)
        self._save_metrics_json(test_metrics, threshold)

        # ── Curves ───────────────────────────────────────────────────
        self._plot_precision_recall_curve(labels, probs, threshold)
        self._plot_roc_curve(labels, probs)
        self._plot_confidence_histogram(probs, labels, threshold)
        self._plot_confusion_matrix(labels, test_preds)

        # ── Export predictions CSV ────────────────────────────────────
        if self.cfg["evaluation"].get("export_predictions", True):
            self._export_predictions(dm, probs, test_preds, labels, threshold)

        return test_metrics

    # ------------------------------------------------------------------
    def _print_report(self, m: dict, threshold: float, labels, preds):
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  BOTTLE INSPECTION MODEL — TEST SET EVALUATION")
        print(f"{sep}")
        print(f"  Decision threshold  : {threshold:.4f}")
        print(f"  Samples             : {len(labels)}")
        print(f"    GOOD   (0)        : {(labels == 0).sum()}")
        print(f"    FAULTY (1)        : {(labels == 1).sum()}")
        print(f"{sep}")
        print(f"  F1 (macro)          : {m['f1']:.4f}")
        print(f"  F1 (FAULTY class)   : {m['f1_faulty']:.4f}  ← primary KPI")
        print(f"  Precision (FAULTY)  : {m['precision_faulty']:.4f}")
        print(f"  Recall    (FAULTY)  : {m['recall_faulty']:.4f}  ← safety metric")
        print(f"  Precision (GOOD)    : {m['precision_good']:.4f}")
        print(f"  Recall    (GOOD)    : {m['recall_good']:.4f}")
        print(f"  Accuracy            : {m['accuracy']:.4f}")
        print(f"  ROC-AUC             : {m['roc_auc']:.4f}")
        print(f"  PR-AUC              : {m['pr_auc']:.4f}")
        print(f"{sep}")
        print(f"  Confusion matrix:")
        print(f"    TP={m['tp']}  FP={m['fp']}")
        print(f"    FN={m['fn']}  TN={m['tn']}")
        print(f"{sep}")
        target_f1 = 0.98
        status = "✓ PASSED" if m["f1_faulty"] >= target_f1 else "✗ BELOW TARGET"
        print(f"  F1 > {target_f1:.0%} target      : {status}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    def _save_metrics_json(self, metrics: dict, threshold: float):
        payload = {"threshold": threshold, **{k: round(float(v), 6) for k, v in metrics.items()}}
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Metrics saved → %s", path)

    def _save_classification_report(self, labels, preds):
        report = classification_report(
            labels, preds, target_names=["GOOD", "FAULTY"], digits=4
        )
        path = self.output_dir / "classification_report.txt"
        with open(path, "w") as f:
            f.write(report)
        print(report)
        logger.info("Classification report saved → %s", path)

    # ------------------------------------------------------------------
    def _plot_precision_recall_curve(self, labels, probs, threshold):
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
        ax.axvline(
            x=self._recall_at_threshold(labels, probs, threshold),
            color="red", linestyle="--", lw=1.2,
            label=f"Calibrated τ = {threshold:.3f}",
        )
        ax.set_xlabel("Recall (FAULTY)")
        ax.set_ylabel("Precision (FAULTY)")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.output_dir / "precision_recall_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("PR curve → %s", path)

    def _plot_roc_curve(self, labels, probs):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.output_dir / "roc_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve → %s", path)

    def _plot_confidence_histogram(self, probs, labels, threshold):
        """Separate histograms of predicted P(FAULTY) for GOOD vs FAULTY bottles."""
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 50)
        ax.hist(probs[labels == 0], bins=bins, alpha=0.6, color="steelblue",  label="GOOD bottles")
        ax.hist(probs[labels == 1], bins=bins, alpha=0.6, color="firebrick",  label="FAULTY bottles")
        ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"τ = {threshold:.3f}")
        ax.set_xlabel("P(FAULTY)")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.output_dir / "confidence_histogram.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Confidence histogram → %s", path)

    def _plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        classes = ["GOOD", "FAULTY"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
        ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix → %s", path)

    # ------------------------------------------------------------------
    def _export_predictions(self, dm, probs, preds, labels, threshold):
        df = dm.test_ds.df.copy()
        df["prob_faulty"]   = probs
        df["pred_label"]    = preds
        df["pred_class"]    = ["FAULTY" if p == 1 else "GOOD" for p in preds]
        df["true_class"]    = ["FAULTY" if l == 1 else "GOOD" for l in labels]
        df["correct"]       = (preds == labels)
        df["threshold_used"] = threshold

        path = self.output_dir / "predictions.csv"
        df.to_csv(path, index=False)
        logger.info("Predictions exported → %s  (%d rows)", path, len(df))

    @staticmethod
    def _recall_at_threshold(labels, probs, threshold):
        preds = (probs >= threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        return tp / max(tp + fn, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained bottle inspection model")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--output_dir", default="outputs/evaluation")
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
    evaluator.run()
