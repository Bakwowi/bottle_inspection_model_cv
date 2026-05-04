"""
src/data/roi_extractor.py
──────────────────────────
Detects the circular bottle base in a raw camera image and crops the
Region of Interest (ROI) to a tight square around it.

Pipeline for each image
───────────────────────
1. Load raw image (BGR → RGB)
2. Convert to grayscale
3. Apply Gaussian blur to suppress sensor noise
4. Run Hough Circle Transform to locate the bottle base circle
5. If detection fails → fall back to fixed centre/radius from config
6. Mask everything outside the circle (black fill) — removes conveyor belt
7. Crop to bounding square with configurable margin
8. Return square PIL image ready for the augmentation / normalisation pipeline

The ROI extractor is intentionally decoupled from the DataLoader so it can
also be used as a pre-processing step that saves cropped images to disk
(see scripts/preprocess_dataset.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

class ROIConfig:
    """Mirrors the image.roi section of config.yaml."""

    def __init__(self, cfg: dict):
        roi = cfg.get("roi", {})
        self.method: str = roi.get("method", "hough_circle")
        self.fixed_center: Tuple[int, int] = tuple(roi.get("fixed_center", [1024, 1024]))
        self.fixed_radius: int = roi.get("fixed_radius", 900)
        self.hough_dp: float = roi.get("hough_dp", 1.2)
        self.hough_min_dist: int = roi.get("hough_min_dist", 500)
        self.hough_param1: int = roi.get("hough_param1", 100)
        self.hough_param2: int = roi.get("hough_param2", 40)
        self.hough_min_radius: int = roi.get("hough_min_radius", 600)
        self.hough_max_radius: int = roi.get("hough_max_radius", 1000)
        self.margin_px: int = roi.get("margin_px", 20)
        self.fallback_to_fixed: bool = roi.get("fallback_to_fixed", True)
        self.input_size: int = cfg.get("input_size", 224)
        self.clahe_enabled: bool = cfg.get("clahe", {}).get("enabled", True)
        self.clahe_clip: float = cfg.get("clahe", {}).get("clip_limit", 2.0)
        self.clahe_grid: Tuple[int, int] = tuple(
            cfg.get("clahe", {}).get("tile_grid_size", [8, 8])
        )


# ── Main extractor ────────────────────────────────────────────────────────────

class ROIExtractor:
    """
    Extracts the bottle-base ROI from a raw industrial camera image.

    Parameters
    ----------
    config : ROIConfig | dict
        Configuration object or raw dict from the image section of config.yaml.
    """

    def __init__(self, config: ROIConfig | dict):
        if isinstance(config, dict):
            config = ROIConfig(config)
        self.cfg = config
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=tuple(self.cfg.clahe_grid),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str | Path) -> Image.Image:
        """
        Load a raw image from disk and return a square PIL Image of the ROI.

        Parameters
        ----------
        image_path : path to the raw image file

        Returns
        -------
        PIL.Image.Image  — square RGB crop of the bottle base
        """
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.extract_from_array(bgr)

    def extract_from_array(self, bgr: np.ndarray) -> Image.Image:
        """
        Same as extract() but accepts an already-loaded BGR numpy array.
        Useful when images arrive from a frame-grabber without touching disk.
        """
        h, w = bgr.shape[:2]
        cx, cy, radius = self._detect_circle(bgr, h, w)
        cropped_bgr = self._crop_and_mask(bgr, cx, cy, radius)

        # Convert BGR → RGB PIL
        rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # Circle detection
    # ------------------------------------------------------------------

    def _detect_circle(
        self,
        bgr: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[int, int, int]:
        """
        Returns (cx, cy, radius) of the detected bottle base circle.

        Strategy
        ────────
        1. Grayscale + median blur (robust to salt-and-pepper noise from
           the industrial camera).
        2. Hough Circle Transform with parameters from config.
        3. If multiple circles found, pick the one closest to image centre.
        4. On failure, fall back to fixed values if configured.
        """
        if self.cfg.method == "fixed":
            return self.cfg.fixed_center[0], self.cfg.fixed_center[1], self.cfg.fixed_radius

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Median blur is more robust than Gaussian against pixel-level noise
        gray_blur = cv2.medianBlur(gray, 7)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=self.cfg.hough_dp,
            minDist=self.cfg.hough_min_dist,
            param1=self.cfg.hough_param1,
            param2=self.cfg.hough_param2,
            minRadius=self.cfg.hough_min_radius,
            maxRadius=self.cfg.hough_max_radius,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # Pick circle whose centre is closest to the image centre
            img_cx, img_cy = w // 2, h // 2
            best = min(
                circles,
                key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2,
            )
            cx, cy, r = int(best[0]), int(best[1]), int(best[2])
            logger.debug("Hough detected circle: centre=(%d,%d) r=%d", cx, cy, r)
            return cx, cy, r

        # Detection failed
        if self.cfg.fallback_to_fixed:
            logger.warning(
                "Hough circle detection failed — using fixed centre/radius fallback."
            )
            return (
                self.cfg.fixed_center[0],
                self.cfg.fixed_center[1],
                self.cfg.fixed_radius,
            )

        raise RuntimeError(
            "Hough circle detection failed and fallback_to_fixed is disabled."
        )

    # ------------------------------------------------------------------
    # Crop & mask
    # ------------------------------------------------------------------

    def _crop_and_mask(
        self,
        bgr: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> np.ndarray:
        """
        1. Mask everything outside the bottle circle with black pixels.
        2. Crop to the bounding square (circle + margin).
        3. Optionally apply CLAHE for contrast normalisation.
        4. Resize to model input_size × input_size.

        Returns
        -------
        np.ndarray  — BGR, shape (input_size, input_size, 3)
        """
        h, w = bgr.shape[:2]
        margin = self.cfg.margin_px

        # ── Step 1: circular mask ─────────────────────────────────────
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius + margin // 2, 255, thickness=-1)
        masked = cv2.bitwise_and(bgr, bgr, mask=mask)

        # ── Step 2: bounding square crop ─────────────────────────────
        x1 = max(cx - radius - margin, 0)
        y1 = max(cy - radius - margin, 0)
        x2 = min(cx + radius + margin, w)
        y2 = min(cy + radius + margin, h)
        crop = masked[y1:y2, x1:x2]

        # Make it a perfect square (pad shorter axis with black)
        crop = self._pad_to_square(crop)

        # ── Step 3: CLAHE on luminance channel ───────────────────────
        if self.cfg.clahe_enabled:
            crop = self._apply_clahe(crop)

        # ── Step 4: resize to model input size ───────────────────────
        size = self.cfg.input_size
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

        return crop

    @staticmethod
    def _pad_to_square(img: np.ndarray) -> np.ndarray:
        """Pad image to square with black pixels on the shorter axis."""
        h, w = img.shape[:2]
        if h == w:
            return img
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=img.dtype)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = img
        return canvas

    def _apply_clahe(self, bgr: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel in LAB colour space.
        Enhances local contrast without blowing out highlights — critical for
        detecting contamination-light and glass imperfections.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── Batch preprocessing script helper ────────────────────────────────────────

def preprocess_and_save(
    annotation_csv: str,
    output_dir: str,
    config: dict,
    image_col: str = "image_path",
    n_jobs: int = 8,
) -> None:
    """
    Pre-compute and save ROI-cropped images to disk before training.
    Saves loading time during training by caching preprocessed crops.

    Parameters
    ----------
    annotation_csv : path to CSV with at minimum an image_path column
    output_dir     : directory to save cropped images
    config         : image section of config.yaml
    image_col      : column name containing raw image paths
    n_jobs         : number of parallel workers
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    df = pd.read_csv(annotation_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = ROIExtractor(config)
    errors: list[str] = []

    def _process_one(row):
        src = Path(row[image_col])
        dst = out / src.name
        if dst.exists():
            return None  # skip already-processed
        try:
            pil_img = extractor.extract(src)
            pil_img.save(dst)
        except Exception as exc:
            return f"{src}: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_process_one, row): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            err = fut.result()
            if err:
                errors.append(err)

    logger.info("Preprocessing complete. Errors: %d / %d", len(errors), len(df))
    if errors:
        for e in errors[:10]:
            logger.warning("  %s", e)
