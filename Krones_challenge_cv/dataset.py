"""
src/data/dataset.py
────────────────────
PyTorch Dataset and PyTorch-Lightning DataModule for the bottle inspection
binary classification task.

Flow per sample
───────────────
CSV row (image_path, label, area_px)
  → LabelResolver.resolve_label()   →  binary target 0/1
  → ROIExtractor.extract()          →  square PIL Image (bottle base only)
  → Albumentations transforms       →  augmented tensor
  → (normalisation)                 →  model-ready float32 tensor [3, H, W]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.roi_extractor import ROIExtractor
from src.utils.label_resolver import resolve_dataframe

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation factory
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(cfg: dict, split: str) -> A.Compose:
    """
    Return an Albumentations Compose pipeline for the given split.

    split : 'train' | 'val' | 'test'
      - train : full augmentation stack
      - val / test : only resize + normalise
    """
    img_cfg  = cfg["image"]
    aug_cfg  = cfg.get("augmentation", {})
    size     = img_cfg["input_size"]
    mean     = img_cfg["mean"]
    std      = img_cfg["std"]

    # ── val / test  (deterministic) ──────────────────────────────────
    if split != "train":
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # ── train (stochastic) ───────────────────────────────────────────
    transforms: List[A.BasicTransform] = [A.Resize(size, size)]

    # Geometric — rotational symmetry of bottle base → full 180°
    if aug_cfg.get("random_rotation", {}).get("enabled", True):
        transforms.append(
            A.Rotate(
                limit=aug_cfg["random_rotation"].get("degrees", 180),
                p=1.0,
            )
        )

    if aug_cfg.get("horizontal_flip", {}).get("enabled", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", {}).get("enabled", True):
        transforms.append(A.VerticalFlip(p=0.5))

    if aug_cfg.get("random_resized_crop", {}).get("enabled", True):
        rrc = aug_cfg["random_resized_crop"]
        transforms.append(
            A.RandomResizedCrop(
                height=size, width=size,
                scale=tuple(rrc.get("scale", [0.85, 1.0])),
                ratio=tuple(rrc.get("ratio", [0.95, 1.05])),
                p=0.5,
            )
        )

    # Photometric
    if aug_cfg.get("color_jitter", {}).get("enabled", True):
        cj = aug_cfg["color_jitter"]
        transforms.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.02),
                p=0.7,
            )
        )

    if aug_cfg.get("gaussian_blur", {}).get("enabled", True):
        gb = aug_cfg["gaussian_blur"]
        transforms.append(
            A.GaussianBlur(
                blur_limit=gb.get("kernel_size", 3),
                sigma_limit=gb.get("sigma", [0.1, 1.5]),
                p=gb.get("p", 0.3),
            )
        )

    if aug_cfg.get("gaussian_noise", {}).get("enabled", True):
        gn = aug_cfg["gaussian_noise"]
        transforms.append(
            A.GaussNoise(
                var_limit=(gn.get("std", 0.02) * 255) ** 2,
                p=gn.get("p", 0.25),
            )
        )

    if aug_cfg.get("grid_distortion", {}).get("enabled", True):
        gd = aug_cfg["grid_distortion"]
        transforms.append(
            A.GridDistortion(
                num_steps=gd.get("num_steps", 5),
                distort_limit=gd.get("distort_limit", 0.15),
                p=gd.get("p", 0.2),
            )
        )

    # Coarse dropout — simulates small occlusions / dirt on lens
    if aug_cfg.get("coarse_dropout", {}).get("enabled", True):
        cd = aug_cfg["coarse_dropout"]
        transforms.append(
            A.CoarseDropout(
                max_holes=cd.get("max_holes", 4),
                max_height=cd.get("max_height", 24),
                max_width=cd.get("max_width", 24),
                fill_value=0,
                p=cd.get("p", 0.3),
            )
        )

    # Normalise + to tensor — always last
    transforms += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BottleDataset(Dataset):
    """
    Parameters
    ----------
    df          : DataFrame with columns [image_path, binary_label]
    transform   : Albumentations Compose pipeline
    roi_extractor: ROIExtractor instance; if None, images are loaded as-is
                   (assumes they are already pre-cropped to ROI)
    use_cache   : keep loaded PIL images in RAM (only for small datasets)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        roi_extractor: Optional[ROIExtractor] = None,
        use_cache: bool = False,
    ):
        self.df            = df.reset_index(drop=True)
        self.transform     = transform
        self.extractor     = roi_extractor
        self.use_cache     = use_cache
        self._cache: dict  = {}

        if "binary_label" not in self.df.columns:
            raise ValueError("DataFrame must contain 'binary_label' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # ── Load image ────────────────────────────────────────────────
        if self.use_cache and idx in self._cache:
            pil_image = self._cache[idx]
        else:
            if self.extractor is not None:
                pil_image = self.extractor.extract(img_path)
            else:
                pil_image = Image.open(img_path).convert("RGB")

            if self.use_cache:
                self._cache[idx] = pil_image

        # ── Augment ───────────────────────────────────────────────────
        np_image = np.array(pil_image)
        augmented = self.transform(image=np_image)
        tensor = augmented["image"]  # float32 [3, H, W]

        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return tensor, label

    @property
    def labels(self) -> List[int]:
        return self.df["binary_label"].tolist()

    def class_counts(self) -> Tuple[int, int]:
        n_good   = (self.df["binary_label"] == 0).sum()
        n_faulty = (self.df["binary_label"] == 1).sum()
        return int(n_good), int(n_faulty)


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class BottleDataModule(pl.LightningDataModule):
    """
    PyTorch-Lightning DataModule.

    Handles
    -------
    - CSV loading & label resolution
    - Stratified train / val / test split
    - WeightedRandomSampler for class imbalance
    - DataLoader construction
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.data_cfg  = cfg["data"]
        self.img_cfg   = cfg["image"]
        self.train_ds: Optional[BottleDataset] = None
        self.val_ds:   Optional[BottleDataset] = None
        self.test_ds:  Optional[BottleDataset] = None
        self._pos_weight: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        # ── 1. Load annotations ───────────────────────────────────────
        df = pd.read_csv(self.data_cfg["annotation_file"])
        df = resolve_dataframe(df, label_col="label", area_col="area_px")

        # ── 2. Stratified split ───────────────────────────────────────
        ratios = self.data_cfg["split_ratios"]
        val_test_frac = ratios["val"] + ratios["test"]
        val_frac_of_remainder = ratios["val"] / val_test_frac

        train_df, temp_df = train_test_split(
            df,
            test_size=val_test_frac,
            stratify=df["binary_label"],
            random_state=self.cfg["project"]["seed"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1.0 - val_frac_of_remainder,
            stratify=temp_df["binary_label"],
            random_state=self.cfg["project"]["seed"],
        )

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        # ── 3. Compute pos_weight for loss ────────────────────────────
        n_good   = (train_df["binary_label"] == 0).sum()
        n_faulty = (train_df["binary_label"] == 1).sum()
        self._pos_weight = torch.tensor([n_good / max(n_faulty, 1)], dtype=torch.float32)
        logger.info(
            "pos_weight (FAULTY / GOOD ratio inverse) = %.4f  "
            "(GOOD=%d, FAULTY=%d)",
            self._pos_weight.item(), n_good, n_faulty,
        )

        # ── 4. Build ROI extractor (shared across splits) ─────────────
        # If images are already pre-cropped (scripts/preprocess_dataset.py
        # was run), set extractor to None and point image_path to cropped dir.
        extractor = ROIExtractor(self.img_cfg)

        # ── 5. Build datasets ─────────────────────────────────────────
        self.train_ds = BottleDataset(
            train_df,
            transform=build_transforms(self.cfg, "train"),
            roi_extractor=extractor,
        )
        self.val_ds = BottleDataset(
            val_df,
            transform=build_transforms(self.cfg, "val"),
            roi_extractor=extractor,
        )
        self.test_ds = BottleDataset(
            test_df,
            transform=build_transforms(self.cfg, "test"),
            roi_extractor=extractor,
        )

    # ------------------------------------------------------------------
    def _make_weighted_sampler(self, ds: BottleDataset) -> WeightedRandomSampler:
        """
        Over-samples the minority class so each epoch sees a balanced set.
        Uses WeightedRandomSampler (with replacement) rather than class duplication.
        """
        labels = ds.labels
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[l] for l in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        sampler = self._make_weighted_sampler(self.train_ds)
        return DataLoader(
            self.train_ds,
            batch_size=self.data_cfg.get("batch_size", self.cfg["training"]["batch_size"]),
            sampler=sampler,
            num_workers=self.data_cfg["num_workers"],
            pin_memory=self.data_cfg["pin_memory"],
            prefetch_factor=self.data_cfg["prefetch_factor"],
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=self.data_cfg["num_workers"],
            pin_memory=self.data_cfg["pin_memory"],
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=self.data_cfg["num_workers"],
            pin_memory=self.data_cfg["pin_memory"],
            persistent_workers=True,
        )

    @property
    def pos_weight(self) -> Optional[torch.Tensor]:
        """Positive class weight for loss function (computed from training set)."""
        return self._pos_weight
