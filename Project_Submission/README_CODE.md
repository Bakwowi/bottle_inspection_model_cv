# Krones Bottle-Base Inspection — Code Documentation

This document explains **how every part of the code works**, for both notebooks. Nothing is left
out: each stage describes what it does, why it's there, and how it connects to the next stage.

---

## Overview of the approach

We treat bottle-base inspection as a **binary classification** problem (GOOD vs FAULTY) but we
train the network with an extra **auxiliary task**: predicting which of the 26 defect categories
are present. This "multi-task" trick is the core of our method — it forces the network to learn
*defect-specific* features instead of a vague yes/no signal, which substantially improves the main
F1 score. On top of the network we add a small **LightGBM stacker** that combines the network's
outputs into a sharper final decision.

```
image ─▶ ROI crop ─▶ EfficientNetV2-S ─┬─▶ main head  (FAULTY prob) ─┐
                                       └─▶ aux head   (26 defect probs)─┤─▶ LightGBM stacker ─▶ threshold ─▶ GOOD/FAULTY
                                                       bottle type ─────┘
```

Two notebooks:
- **Training notebook** — does EDA, trains the network (3 CV folds) + stacker, saves all artifacts.
- **Evaluation notebook** — loads the saved artifacts, runs inference on the test set, writes
  `submission.csv`, and measures runtime. This is the only notebook the organizers execute.

---

## TRAINING NOTEBOOK — stage by stage

### 0. Imports & reproducibility
We import PyTorch, timm (pretrained models), LightGBM (the stacker), OpenCV (fast image IO/crop),
and scikit-learn (metrics + CV splits). `seed_everything()` fixes all random seeds so the run is
reproducible. `cudnn.benchmark = True` lets cuDNN pick the fastest convolution kernels for our
fixed image size.

### 1. Configuration (`CFG`)
A single dictionary holds every setting. Key entries:
- `backbone = tf_efficientnetv2_s.in21k_ft_in1k` — EfficientNetV2-S pretrained on ImageNet-21k then
  1k. Strong features, small and fast (good for the efficiency score).
- `img_size = 448` — bottle bases have fine detail; 448px keeps small defects visible.
- `in_chans = 1` — the photos are grayscale, so we use 1 channel. timm averages the pretrained RGB
  stem filters to accept 1 channel. Cleaner and slightly faster than feeding 3 identical channels.
- `aux_weight = 0.4` — how much the auxiliary 26-category loss counts vs the main loss.
- `use_advanced_arch`, `use_tta` — optional upgrades (off by default), explained below.
- `lgb_params` — LightGBM hyperparameters for the stacker.

### 2. Load labels and metadata
`train.csv` gives the binary `target` per image. `bottletypes.csv` (if present) gives the bottle
model, used for (a) fairer CV stratification and (b) a stacker feature.

### 3. Exploratory Data Analysis (EDA)
- **Class balance**: bar chart of GOOD vs FAULTY and images per bottle type. Shows the dataset is
  imbalanced (more FAULTY), which is why we use `pos_weight` in the loss.
- **Defect categories**: parses `train_annotations.json`, counts each defect type, plots a
  horizontal bar chart. Reveals strong imbalance — a few defect types are very rare. This motivates
  the auxiliary head (gives rare defects their own supervision signal).
- **Image sizes**: confirms images share a consistent resolution, so our crop+resize is uniform.
- **Example images**: shows GOOD vs FAULTY bottles. Observation: defects are often small/faint
  relative to the whole base — motivating the ROI crop (zoom in) and the aux head (learn subtle
  defect appearances).

### 4. ROI boxes and auxiliary targets
`train_annotations.json` is a COCO file. **Category id 22 is the ROI** (the base region); all other
categories are defect types.
- We build `tr_roi`: filename → ROI bounding box `[x, y, w, h]`.
- We build `aux_targets`: filename → 26-dim 0/1 vector marking which defect categories appear in
  that image. These are the auxiliary labels.

### 5. ROI crop + Dataset
- `crop_roi()` makes a **square** crop centred on the ROI (so resizing doesn't distort the round
  base), adds a 6% margin, and **reflect-pads** if the square extends past the image edge. It uses
  `INTER_AREA` when shrinking (sharper) and `INTER_LINEAR` when enlarging.
- `FALLBACK_ROI` = median ROI box, used for any image missing an ROI so it still gets a sensible
  crop.
- `BottleDataset` loads a grayscale image, crops it to the ROI, optionally augments (flips,
  90° rotations, small brightness jitter — all label-preserving because bottle bases are
  rotationally symmetric), normalizes to the grayscale mean/std, and returns
  `(image_tensor, target, aux_vector)`.

### 6. The model — `KronesNet`
A shared EfficientNetV2-S backbone feeds **two heads**:
- `head` → 1 logit: the FAULTY/GOOD prediction (the scored task).
- `aux_head` → 26 logits: which defect categories are present (the helper task).

Training both forces the backbone features to encode per-defect information. At inference we only
need the main head, but the aux probabilities also feed the stacker.

Optional (`use_advanced_arch=True`):
- **GeMPooling** replaces global average pooling. GeM raises activations to a power `p` before
  averaging, emphasising peak responses — so a small, localized defect activation isn't averaged
  away. Done in fp32 with `p` clamped to [1,5] for numerical safety under mixed precision.
- **SEGate** (squeeze-excite) learns a per-channel importance weight (channel attention).

### 7. Loss, threshold search, prediction helper
- **Loss** = `main_BCE(pos_weight) + aux_weight * aux_BCE`. `pos_weight = neg/pos` makes the loss
  pay more attention to the minority class.
- `best_f1_threshold()` sweeps a fine grid (0.05→0.95, step 0.0025) for the threshold that
  maximises F1.
- `predict_feats()` runs a model and returns both the main probability and the 26 aux probabilities
  (optionally averaging TTA views). These outputs are what the stacker consumes.

### 8. Cross-validation folds
`StratifiedKFold` on the combined `target × bottle_type` key (rare combos fall back to label-only),
producing 3 folds with similar composition. We train on 2 folds, validate on the 3rd, rotating —
giving **out-of-fold (OOF)** predictions for every training image (predictions from a model that
never saw that image). `AUX_MAT` is the 26-category target matrix aligned to row order.

### 9. Train folds + collect OOF
For each fold: train the network with the multi-task loss, keep the best-validation-F1 epoch
(saved as `model_fold{k}.pt`), then run that best model over the held-out fold to record OOF main +
aux probabilities. We use AdamW + cosine LR schedule + mixed precision (AMP). After all folds, OOF
arrays cover the whole training set; we save `oof.csv` and `oof_aux.npy`.

### 10. The LightGBM stacker
Stacker input per image = `[ logit(main prob), 26 aux probs, bottle-type one-hot ]`.
- Trained **fold-safe**: each fold's OOF rows are predicted by trees fit on the *other* folds, so
  there's no leakage when we measure its OOF F1.
- We compare three options on OOF F1: **raw** (network prob alone), **stack** (stacker output),
  **blend** (50/50). We keep whichever wins (`mode`), with its best threshold.
- A final stacker is retrained on all OOF rows for test use and saved as `stacker.txt`.
- All decisions (mode, threshold, backbone, image size, etc.) are saved to `best_threshold.json`
  so the evaluation notebook stays perfectly in sync.

### 11. Result analysis
Confusion matrix + score-distribution histogram. The histogram shows GOOD scores clustered low and
FAULTY scores clustered high with a clean gap — i.e. the model separates the classes well, and the
threshold sits in the valley between them.

### Artifacts produced
`model_fold0.pt`, `model_fold1.pt`, `model_fold2.pt`, `stacker.txt`, `best_threshold.json`,
`oof.csv`, `oof_aux.npy` — all in `/kaggle/working/`.

---

## EVALUATION NOTEBOOK — stage by stage

This notebook is **self-contained** and runs top-to-bottom in a clean environment.

### 0–1. Imports & config loading
Loads `best_threshold.json` from the attached model dataset. Every setting (backbone, image size,
channels, normalization, threshold, mode, fold filenames) comes from that file — nothing is
hard-coded twice, so the evaluation can't drift from how the model was trained.

### 2. Model definition
Re-declares the identical `KronesNet` (and GeM/SE if used) so the saved weights load correctly.

### 3. ROI crop + test dataset
Same `crop_roi` and preprocessing as training. Reads test ROI boxes from
`test_annotations_roi_only.json` (the test set ships ROI-only annotations), with a median fallback.

### 4. Load test list + models + stacker
Builds the test image list from the directory (sorted deterministically), loads all fold models,
loads the LightGBM stacker, and the bottle-type map.

### 5. Inference function
For each batch: run every fold model (averaging their main + aux probabilities, plus TTA views if
enabled), build the stacker features `[logit(main), aux, bottle-type one-hot]`, get the stacker
probability, and return the final probability under the chosen `mode` (raw/stack/blend).

### 6. Submission
Thresholds the probabilities and writes `submission.csv` (`image_id, target`).

### 7. ⏱️ Runtime measurement
The efficiency metric. Warms up first (so one-time CUDA/cuDNN setup isn't counted), then times the
**full pipeline** (load + crop + preprocess + all folds + stacker) over the whole test set. Prints
total time, per-image ms, throughput (img/s), and headroom vs the 70,000 bottles/hour line speed.

---

## Why each design choice helps the score (50% F1 / 30% efficiency / 20% insight)

| Choice | Helps |
|---|---|
| Aux 26-category head | **F1** — richer supervision → defect-aware features (the main lever) |
| LightGBM stacker | **F1** — learned combination of per-category evidence |
| ROI crop + 448px | **F1** — small/faint defects get more pixels |
| EfficientNetV2-S, grayscale 1-channel | **Efficiency** — small fast backbone, lean input |
| GeM (optional) | **F1** — small-defect signal survives pooling |
| Single-model option | **Efficiency** — drop the fold ensemble if speed matters more than the last F1 |
