"""
scripts/preprocess_dataset.py
──────────────────────────────
One-time script: runs ROI extraction on all raw images and saves the
cropped, CLAHE-enhanced, resized crops to disk.

Benefits
────────
- Eliminates per-epoch ROI computation → faster training DataLoader
- Lets you visually inspect the crops before training
- Detects bad images / failed detections upfront

Usage
─────
  python scripts/preprocess_dataset.py \
      --config configs/config.yaml \
      --annotation_csv data/annotations.csv \
      --output_dir data/processed \
      --n_jobs 8

After running, update config.yaml:
  data.processed_dir: data/processed
  (and set roi_extractor=None in BottleDataset by pointing image_path to processed dir)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.roi_extractor import ROIExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def run(
    annotation_csv: str,
    output_dir: str,
    config_path: str,
    image_col: str = "image_path",
    n_jobs: int = 8,
    skip_existing: bool = True,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df  = pd.read_csv(annotation_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = ROIExtractor(cfg["image"])

    errors: list[tuple[str, str]] = []

    def _process(row):
        src = Path(row[image_col])
        dst = out / src.name
        if skip_existing and dst.exists():
            return None, None
        try:
            pil_img = extractor.extract(src)
            pil_img.save(dst)
            return None, None
        except Exception as exc:
            return str(src), str(exc)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_process, row): i for i, row in df.iterrows()}
        with tqdm(total=len(futures), desc="Preprocessing ROIs") as pbar:
            for fut in as_completed(futures):
                src_path, err_msg = fut.result()
                if src_path:
                    errors.append((src_path, err_msg))
                pbar.update(1)

    logger.info("Done. Processed: %d | Errors: %d", len(df) - len(errors), len(errors))
    if errors:
        err_path = out / "preprocessing_errors.txt"
        with open(err_path, "w") as f:
            for src, msg in errors:
                f.write(f"{src}\t{msg}\n")
        logger.warning("Error log saved → %s", err_path)

    # Update annotation CSV to point to processed directory
    new_df = df.copy()
    new_df[image_col] = new_df[image_col].apply(
        lambda p: str(out / Path(p).name)
    )
    new_csv = Path(annotation_csv).parent / "annotations_processed.csv"
    new_df.to_csv(new_csv, index=False)
    logger.info("Updated annotation CSV saved → %s", new_csv)
    logger.info("Update config.yaml: data.annotation_file: %s", new_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--annotation_csv", default="data/annotations.csv")
    parser.add_argument("--output_dir",     default="data/processed")
    parser.add_argument("--image_col",      default="image_path")
    parser.add_argument("--n_jobs",         type=int, default=8)
    parser.add_argument("--no_skip",        action="store_true",
                        help="Re-process even if output file exists")
    args = parser.parse_args()

    run(
        annotation_csv=args.annotation_csv,
        output_dir=args.output_dir,
        config_path=args.config,
        image_col=args.image_col,
        n_jobs=args.n_jobs,
        skip_existing=not args.no_skip,
    )
