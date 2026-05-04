# Bottle Inspection — Binary Classifier

High-performance CNN classifier that determines whether a bottle is **usable (GOOD)** or **reject (FAULTY)** from an image of its base, targeting **F1 > 98%**.

---

## Project Structure

```
bottle_inspection/
├── configs/
│   └── config.yaml              ← all hyperparameters & label definitions
├── src/
│   ├── utils/
│   │   └── label_resolver.py    ← raw label + area → binary 0/1
│   ├── data/
│   │   ├── roi_extractor.py     ← Hough circle → crop → CLAHE → resize
│   │   └── dataset.py           ← Dataset, DataModule, augmentation factory
│   ├── models/
│   │   ├── classifier.py        ← EfficientNet-B0 + custom head
│   │   └── losses.py            ← Focal Loss + loss factory
│   ├── training/
│   │   ├── lightning_module.py  ← LightningModule: 2-phase training + metrics
│   │   └── hparam_tuning.py     ← Optuna hyperparameter search
│   └── evaluation/
│       └── evaluator.py         ← threshold calibration + full metrics + plots
├── scripts/
│   ├── preprocess_dataset.py    ← pre-crop all images to disk (run once)
│   └── train.py                 ← main training entry point
└── requirements.txt
```

---

## Full Pipeline Architecture

### Stage 1 — Label Injection

**File:** `src/utils/label_resolver.py`

Raw dataset annotations carry a `label` string and an `area_px` float. The resolver applies three-tier logic before any model sees the data:

| Tier | Labels | Rule |
|------|--------|------|
| Always GOOD | `embossing`, `foam_residue`, `no_fault`, `water_drop` | → **0** |
| Conditionally FAULTY | `air_bubble`, `chip`, `contamination_light`, `glass_imperfection`, `scuffing`, `scuffing_heavy` | → **1** only if `area_px > threshold` |
| Always FAULTY | `break_crack`, `circlip`, `contamination_dark`, `crown_cap`, `foil_semitransparent`, `foreign_object_*`, `glass_shard`, `insect`, `label`, `liquid`, `mold`, `no_base_visible`, `paint_residue`, `straw`, `yeast_residue` | → **1** |

```python
from src.utils.label_resolver import resolve_dataframe
df = resolve_dataframe(df, label_col="label", area_col="area_px")
# adds column: binary_label  {0=GOOD, 1=FAULTY}
```

---

### Stage 2 — ROI Extraction

**File:** `src/data/roi_extractor.py`

Each raw camera image contains the bottle base surrounded by the conveyor belt and background. The extractor isolates only the bottle:

```
Raw image (e.g. 2048×2048)
  │
  ▼
Grayscale + Median Blur (7×7)
  │
  ▼
Hough Circle Transform
  → detects circular bottle base
  → picks circle closest to image centre
  → fallback to fixed centre/radius if detection fails
  │
  ▼
Circular mask (everything outside → black)
  │
  ▼
Square bounding crop (circle bbox + margin_px)
  │
  ▼
Pad shorter axis to square
  │
  ▼
CLAHE on L channel (LAB colour space)
  → enhances local contrast
  → critical for contamination_light and glass_imperfection
  │
  ▼
Resize to 224×224 (INTER_AREA)
  │
  ▼
PIL Image (RGB)  ← ready for transforms
```

**Pre-process once, train faster:**
```bash
python scripts/preprocess_dataset.py \
    --config configs/config.yaml \
    --annotation_csv data/annotations.csv \
    --output_dir data/processed \
    --n_jobs 8
```

---

### Stage 3 — Augmentation (training only)

**File:** `src/data/dataset.py` → `build_transforms()`

All transforms are applied via Albumentations. Val/test only get resize + normalise.

| Transform | Parameters | Rationale |
|-----------|-----------|-----------|
| `Rotate` | ±180° | Bottle bases are rotationally symmetric |
| `HorizontalFlip` | p=0.5 | Symmetry |
| `VerticalFlip` | p=0.5 | Symmetry |
| `RandomResizedCrop` | scale [0.85–1.0] | Slight scale variation |
| `ColorJitter` | brightness ±0.2, contrast ±0.2 | Lighting variation |
| `GaussianBlur` | σ [0.1–1.5], p=0.3 | Lens defocus simulation |
| `GaussNoise` | std 0.02, p=0.25 | Sensor noise |
| `GridDistortion` | limit 0.15, p=0.2 | Lens distortion simulation |
| `CoarseDropout` | 4 holes ≤ 24px, p=0.3 | Occlusion / dirt on lens |
| `Normalize` | ImageNet μ/σ | Matches pretrained weights |

---

### Stage 4 — Model Architecture

**File:** `src/models/classifier.py`

```
Input: [B, 3, 224, 224]
  │
  ▼
EfficientNet-B0 (pretrained ImageNet)
  └─ Stem + 7 MBConv blocks
  └─ Global Average Pooling
  → [B, 1280]
  │
  ▼
Dropout(0.4)
  │
  ▼
Linear(1280 → 256) + BatchNorm1d + ReLU
  │
  ▼
Dropout(0.2)
  │
  ▼
Linear(256 → 1)
  │
  ▼
[B, 1]  logits  (Sigmoid applied at inference → P(FAULTY))
```

**Why EfficientNet-B0?**
- ~5.3M parameters — small enough for <5ms inference with INT8 TensorRT
- Compound scaling: balanced depth/width/resolution for the available compute
- Strong ImageNet pretraining transfers well to circular object inspection

---

### Stage 5 — Loss Function

**File:** `src/models/losses.py`

**Focal Loss** (Lin et al. 2017):

```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
```

- `γ = 2.0` — focuses training on hard borderline samples (e.g. small conditional defects near the threshold)
- `α = 0.75` — weights FAULTY class 3× more than GOOD; tune to your actual imbalance ratio
- Down-weights easy negatives (obvious good bottles) automatically

---

### Stage 6 — Two-Phase Training

**File:** `src/training/lightning_module.py`

```
Phase 1  (epochs 0–7):
  Backbone FROZEN
  Head only trained
  LR = 1e-3  (high — fast head warmup)
  Avoids destroying pretrained features

Phase 2  (epochs 8–49):
  Last 3 EfficientNet blocks UNFROZEN
  Full fine-tune
  LR = 1e-4 → cosine decay → 1e-7
  3-epoch linear warmup at start of phase 2
```

**Scheduler:** linear warmup → flat (phase 1) → cosine annealing (phase 2)

**Optimiser:** AdamW (`lr=1e-4`, `weight_decay=1e-4`, `β=(0.9,0.999)`)

**Class imbalance:** `WeightedRandomSampler` in training DataLoader — each epoch sees a balanced draw regardless of the true class ratio.

---

### Stage 7 — Hyperparameter Tuning

**File:** `src/training/hparam_tuning.py`

Uses **Optuna** with TPE sampler and Median pruner:

```bash
python -m src.training.hparam_tuning \
    --config configs/config.yaml \
    --n_trials 50 \
    --n_epochs 15 \
    --study_name bottle_v1 \
    --storage sqlite:///hparam.db
```

**Search space:**

| Parameter | Type | Range |
|-----------|------|-------|
| `backbone` | categorical | efficientnet_b0, mobilenet_v3_small |
| `lr_phase1` | log-uniform | [1e-4, 5e-3] |
| `lr_phase2` | log-uniform | [1e-6, 5e-4] |
| `focal_gamma` | float | [1.0, 4.0] |
| `focal_alpha` | float | [0.5, 0.9] |
| `dropout1` | float | [0.2, 0.6] |
| `dropout2` | float | [0.1, 0.4] |
| `hidden_dim` | categorical | [128, 256, 512] |
| `weight_decay` | log-uniform | [1e-5, 1e-3] |
| `batch_size` | categorical | [32, 64, 128] |
| `rotation_degrees` | categorical | [90, 180] |

**Safety constraint:** Any trial where `val_recall_faulty < 0.99` returns score 0.0.

---

### Stage 8 — Threshold Calibration & Evaluation

**File:** `src/evaluation/evaluator.py`

The default 0.5 sigmoid threshold is rarely optimal. After training:

1. Run inference on the **validation set** to collect `P(FAULTY)` scores
2. Sweep all candidate thresholds from the precision-recall curve
3. For each threshold: check `recall_faulty ≥ 0.99` (safety gate)
4. Among passing thresholds: pick `argmax(F1)`
5. Apply calibrated threshold to **test set** — report final metrics

**Metrics reported:**
- F1 (macro + FAULTY class)
- Precision & Recall (both classes)
- Accuracy
- ROC-AUC
- PR-AUC
- Confusion matrix
- Full sklearn classification report

**Outputs saved:**
```
outputs/evaluation/
├── metrics.json
├── classification_report.txt
├── predictions.csv
├── precision_recall_curve.png
├── roc_curve.png
├── confidence_histogram.png
└── confusion_matrix.png
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare annotations CSV
# Required columns: image_path, label, area_px
# label values must match the names in config.yaml

# 3. Pre-process ROIs (recommended — run once)
python scripts/preprocess_dataset.py \
    --annotation_csv data/annotations.csv \
    --output_dir data/processed

# 4. (Optional) Hyperparameter search
python -m src.training.hparam_tuning \
    --n_trials 50 --n_epochs 15

# 5. Train
python scripts/train.py --config configs/config.yaml

# 6. Evaluate
python -m src.evaluation.evaluator \
    --checkpoint checkpoints/best.ckpt \
    --config configs/config.yaml
```

---

## Annotation CSV Format

```csv
image_path,label,area_px
data/raw/bottle_001.png,no_fault,
data/raw/bottle_002.png,air_bubble,620
data/raw/bottle_003.png,glass_shard,
data/raw/bottle_004.png,water_drop,45
data/raw/bottle_005.png,scuffing,82000
```

- `area_px` is required only for conditional labels; leave blank otherwise.
- Label strings are case-insensitive and spaces/hyphens are normalised to underscores.
