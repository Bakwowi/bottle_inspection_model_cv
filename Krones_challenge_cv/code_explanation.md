 # Bottle Inspection Notebook — Complete Code Explanation Guide

> This document explains every function, every parameter, and every CFG key in the notebook.
> Written assuming no prior knowledge of neural networks or CNNs.

---

## First: What is a Neural Network? (The mental model you need)

Think of a neural network as a very large mathematical function that takes an image as input
and produces a number as output. For our project:

```
Input:  a 224×224 pixel image of a bottle base
Output: one number between 0 and 1
        → close to 0 means "this bottle is GOOD"
        → close to 1 means "this bottle is FAULTY"
```

The network learns by being shown thousands of examples with known answers (labels),
comparing its output to the correct answer, measuring how wrong it was (the loss),
and nudging its internal numbers (weights) slightly to be less wrong next time.
This process of nudging is called backpropagation, and one full pass through all
training images is called an epoch.

A CNN (Convolutional Neural Network) is a type of neural network specifically designed
for images. It works by scanning small filters across the image to detect patterns —
first simple things like edges and corners, then progressively more complex things like
textures, shapes, and eventually full objects like glass shards or water droplets.

---

## SECTION 1 — The CFG Dictionary

The CFG dictionary is the single control panel for the entire project.
Every important number lives here. You never need to search through code to change a setting.

```python
CFG = { ... }
```

### Paths

```python
'annotation_csv':  '/kaggle/input/your-dataset/annotations.csv'
```
Where your CSV file lives. This CSV has one row per image with at minimum
columns for the image path, the raw defect label (e.g. "air_bubble"), and
the defect area in pixels.

```python
'processed_dir': '/kaggle/working/processed'
```
Where pre-cropped bottle base images will be saved after the ROI extraction step runs.
Running ROI extraction once and saving results here means training is much faster
because images are ready to load directly without re-running circle detection each epoch.

```python
'output_dir': '/kaggle/working/outputs'
```
Where all results are saved after training — prediction CSV, confusion matrix image,
ROC curve image, metrics JSON.

```python
'checkpoint_path': '/kaggle/working/best_model.pt'
```
Where the best version of the model is saved during training. Every time the model
achieves a new best validation F1 score, its weights are saved here. If training
crashes or you need to continue later, you load from this file.

---

### Label columns

```python
'label_col':  'label'
'area_col':   'area_px'
'image_col':  'image_path'
```
These tell every function which column names to look for in your CSV.
If your CSV uses different column names (e.g. "defect_type" instead of "label"),
change these strings here rather than hunting through the code.

---

### Image / ROI settings

```python
'input_size': 224
```
The size in pixels of the image that gets fed into the neural network.
The model expects a square image of this size. 224×224 is the standard size
for EfficientNet — it was designed and pretrained at this resolution.
All images, regardless of their original size, are resized to 224×224.

```python
'roi_method': 'hough'   # or 'fixed'
```
How to find the circular bottle base in the raw image.
'hough' uses the Hough Circle Transform algorithm to detect the circle automatically.
'fixed' uses the manually-specified centre and radius below.
If using COCO annotations (the improved version), this setting is ignored.

```python
'fixed_cx': 1024
'fixed_cy': 1024
'fixed_radius': 900
```
When roi_method is 'fixed', or when Hough detection fails, the ROI is defined
as a circle centred at pixel coordinate (1024, 1024) with radius 900 pixels.
These are fallback values — measure them from a representative bottle image
using any image editor (GIMP, Photoshop, etc.).

```python
'hough_dp': 1.2
```
Inverse ratio of the accumulator resolution to the image resolution.
Value of 1 means the accumulator has the same resolution as the image.
1.2 means slightly lower resolution — faster but slightly less precise.
In practice, keep this between 1.0 and 2.0.

```python
'hough_min_dist': 500
```
Minimum distance in pixels between the centres of two detected circles.
Set this to roughly the diameter of the smallest bottle you expect, so the
algorithm doesn't detect two overlapping circles for the same bottle.

```python
'hough_param1': 100
```
The upper threshold for the Canny edge detector that runs inside Hough.
Higher values mean only strong, clear edges trigger detections.
Lower values detect more edges including noisy ones.
Typical range: 50–200.

```python
'hough_param2': 40
```
The accumulator threshold — how many edge points must "vote" for a circle
before it's accepted as a detection. Lower values find more circles (including
false ones). Higher values are more conservative. Start at 40 and increase
if you're getting false circle detections.

```python
'hough_min_r': 600
'hough_max_r': 1000
```
The minimum and maximum radius of circles the algorithm will look for, in pixels.
Set these to bracket the actual bottle base radius in your images.
If your bottles have radius ~900px, setting min=600 and max=1000 prevents
detecting irrelevant small circles (bottle cap reflections) or huge ones.

```python
'roi_margin': 20
```
Extra pixels added around the detected circle before cropping.
A margin of 20 means the crop extends 20 pixels beyond the circle edge.
This ensures the very edge of the bottle base is included and not cut off.

```python
'clahe_clip': 2.0
```
Controls how aggressively CLAHE enhances local contrast.
Higher values = more contrast enhancement = better visibility of subtle defects,
but also amplifies camera sensor noise. Values between 1.0 and 4.0 are typical.
2.0 is a safe default for industrial inspection cameras.

```python
'clahe_grid': (8, 8)
```
CLAHE divides the image into a grid of tiles and equalises each tile independently.
(8, 8) means the image is divided into an 8×8 grid = 64 tiles.
Smaller tiles give more localised contrast enhancement, better for small defects.
Larger tiles give more global enhancement. (8, 8) is the standard default.

```python
'img_mean': [0.485, 0.456, 0.406]
'img_std':  [0.229, 0.224, 0.225]
```
These are the average pixel values (mean) and spread (std) of the ImageNet dataset,
measured across millions of natural images, on a 0–1 scale for R, G, B channels.
EfficientNet was trained on ImageNet with these statistics, so every image fed to
it must be normalised using these exact same numbers. Without this, the pretrained
features don't work because the model expects input in a specific range.

Normalisation formula per pixel: normalised = (pixel_value / 255 - mean) / std

---

### Data splits

```python
'train_frac': 0.70
'val_frac':   0.15
# test_frac is implicitly 0.15
```
What fraction of your total dataset goes to each purpose:
- Train set (70%): the model learns from these images
- Validation set (15%): used during training to check performance and save the best checkpoint. The model never learns from these.
- Test set (15%): completely hidden until the very end. Final reported F1 comes from here.

Why three sets? If you only used train and test, you might accidentally tune
your model to the test set by trying many configurations. The validation set
absorbs all the tuning so the test set remains a genuine unseen evaluation.

---

### Model settings

```python
'backbone': 'efficientnet_b0'
```
Which pretrained neural network to use as the feature extractor.
EfficientNet-B0 has ~5.3 million parameters and was trained on 1.28 million
ImageNet images. It already knows how to detect edges, textures, shapes, and
complex visual patterns. We keep its knowledge and adapt it to bottle inspection.
Alternatives: 'mobilenet_v3_small' (faster, slightly less accurate),
'resnet50' (larger, more accurate, slower).

```python
'dropout1': 0.4
```
After the backbone extracts features, we apply Dropout before the first fully
connected layer. Dropout randomly sets 40% of the feature values to zero during
training. This sounds destructive but it forces the model to not rely on any
single feature too heavily — a form of regularisation that prevents overfitting.
At inference (prediction time) dropout is turned off.

```python
'hidden_dim': 256
```
The number of neurons in the intermediate layer of the classification head.
The head goes: 1280 features → 256 neurons → 1 output.
The 1280→256 compression forces the model to find the most important features
for the decision. Larger values (512) give more capacity, smaller (128) are faster.

```python
'dropout2': 0.2
```
A second, lighter dropout (20%) applied after the 256-neuron layer.
Provides secondary regularisation on the compressed feature representation.

---

### Loss settings

```python
'focal_gamma': 2.0
```
The focusing exponent in Focal Loss. Controls how much the loss down-weights
easy examples. gamma=0 makes it identical to standard cross-entropy.
gamma=2 means: if the model is 90% confident on an easy bottle, its contribution
to the loss is multiplied by (1-0.9)^2 = 0.01 — almost ignored.
A hard borderline bottle at 50% confidence contributes (1-0.5)^2 = 0.25 — full weight.
Range: 0.5–5. Start at 2.0. Increase if too many easy examples dominate training.

```python
'focal_alpha': 0.42
```
The class balance weight. Specifically weights how much the model cares about
the FAULTY class relative to GOOD.
0.42 means: FAULTY errors contribute alpha=0.42 weight,
            GOOD errors contribute (1-0.42)=0.58 weight.
This value should roughly match the GOOD class frequency (14729/35342 = 0.417).
HIGHER values → model becomes more aggressive about predicting FAULTY (more recall, less precision)
LOWER values  → model becomes more conservative about FAULTY (less recall, more precision)
This is the single most impactful hyperparameter in your project.

---

### Training settings

```python
'total_epochs': 50
```
How many times to show the entire training set to the model.
One epoch = one full pass through all training images.
More epochs = more learning, but eventually the model stops improving (or overfits).
Early stopping (below) prevents wasting time on unprofitable epochs.

```python
'phase1_epochs': 8
```
How many epochs to train with the backbone frozen (only the head learns).
During these epochs, the randomly-initialised head learns the basic task
without destroying the backbone's pretrained features.

```python
'phase2_epochs': 42
```
Epochs for fine-tuning with some backbone layers unfrozen.
The backbone adapts its features to bottle-specific patterns.

```python
'warmup_epochs': 3
```
For the first 3 epochs, the learning rate gradually increases from near-zero
to the target learning rate (linear warmup). Without warmup, a large learning
rate at epoch 0 on a randomly-initialised head can send the model in a completely
wrong direction from which it never recovers.

```python
'batch_size': 64
```
How many images the model sees in one forward-backward pass before updating weights.
Larger batches: more stable gradient estimates, faster GPU utilisation.
Smaller batches: more frequent updates, sometimes better generalisation.
64 is a good default for EfficientNet-B0 on a modern GPU.
Reduce to 32 if you run out of GPU memory.

```python
'lr_phase1': 1e-3
```
Learning rate during phase 1 (head only training).
Learning rate controls how large a step the model takes toward lower loss.
Too high: model overshoots the minimum and diverges.
Too low: model learns extremely slowly or gets stuck.
1e-3 (= 0.001) is a standard starting point for training a fresh head.

```python
'lr_phase2': 1e-4
```
Learning rate during phase 2 (fine-tuning backbone).
Must be lower than phase 1 because we're adjusting pretrained weights
that are already close to good values — small steps prevent destroying them.

```python
'min_lr': 1e-7
```
The minimum learning rate the cosine annealing scheduler is allowed to reach.
The LR decays smoothly from lr_phase2 down to min_lr over phase 2.
Acts as a floor to prevent the LR from reaching zero (which would completely
stop learning).

```python
'weight_decay': 1e-4
```
L2 regularisation strength. Adds a small penalty to the loss proportional
to the square of each weight, encouraging weights to stay small.
Prevents any single weight from growing so large it dominates decisions.
1e-4 (= 0.0001) is a standard value for AdamW.

```python
'grad_clip': 1.0
```
Maximum allowed gradient norm. If the gradient during backpropagation exceeds
this value, it is scaled down to exactly 1.0.
Prevents "gradient explosion" — a situation where a bad batch causes enormous
weight updates that destabilise the entire model.

```python
'amp': True
```
Automatic Mixed Precision. When True, uses 16-bit floating point (fp16) for
most computations instead of 32-bit (fp32). This is mathematically nearly
identical but uses half the memory and runs ~2× faster on modern NVIDIA GPUs
(which have dedicated fp16 hardware called Tensor Cores).

```python
'num_workers': 4
```
How many CPU processes load and preprocess images in parallel while the GPU
is running computations. 4 workers means 4 images are being prepared for
the next batch while the current batch is training. Set to 0 if you encounter
multiprocessing errors on Windows.

---

### Early stopping

```python
'early_stop_patience': 10
```
If the validation F1 doesn't improve for 10 consecutive epochs, stop training.
Prevents wasting compute on epochs that aren't helping.
Also prevents overfitting — if the model has converged, training longer
often makes it worse on unseen data.

```python
'early_stop_min_delta': 5e-4
```
The minimum improvement in validation F1 that counts as "getting better".
5e-4 = 0.0005. An improvement of less than 0.05% F1 is treated as flat.
Without this, a tiny noise-driven improvement resets the patience counter.

---

### Threshold calibration

```python
'min_recall_faulty': 0.990
```
Safety constraint: the calibrated threshold must achieve at least 99% recall
on FAULTY bottles. This means at most 1 in 100 faulty bottles is allowed to
be called GOOD (a false negative). This constraint is prioritised over F1 —
if you have to choose between a threshold that maximises F1 but misses 3%
of faulty bottles, or one with slightly lower F1 that misses only 0.5%,
this setting forces the safer choice.

```python
'default_threshold': 0.5
```
Starting threshold used during training for computing validation F1.
This is not the final threshold — the real threshold is calibrated on the
validation set after training using calibrate_threshold(). 0.5 is used
during training purely for monitoring progress.

---

### Optuna settings

```python
'optuna_n_trials': 30
```
How many different hyperparameter combinations Optuna will try.
Each trial trains a model for optuna_n_epochs and reports validation F1.
More trials = better search but more compute time.
30 trials with 12 epochs each = 360 epochs total of search compute.

```python
'optuna_n_epochs': 12
```
How many epochs each Optuna trial trains for.
Short enough to be fast (doesn't need to converge fully),
long enough to differentiate good configurations from bad ones.

---

## SECTION 2 — Label Injection Functions

### `normalise_label(raw)`

```python
def normalise_label(raw: str) -> str:
```

**What it does:** Converts a raw label string into a consistent format for lookup.

**Why it's needed:** Your CSV might have labels written as "Break/Crack",
"break_crack", "Break Crack", or "BREAK-CRACK". All of these mean the same thing.
This function converts all of them to the same key: "break_crack".

**How it works:**
1. Strip whitespace from both ends
2. Convert to lowercase
3. Replace spaces, hyphens, and slashes with underscores
4. Collapse multiple consecutive underscores into one

**Parameters:**
- `raw` — any raw label string from the CSV

**Returns:** A normalised lowercase string with underscores as separators.

---

### `resolve_single_label(label, area_px)`

```python
def resolve_single_label(label: str, area_px: Optional[float] = None) -> int:
```

**What it does:** Converts one raw label + area measurement into 0 (GOOD) or 1 (FAULTY).

**Why it's needed:** Your dataset has 25+ label types but the model only needs to
output one binary decision. This function is where all domain expert knowledge
lives — the rules about which defects are acceptable and which aren't.

**How it works (three-tier check, in order):**

Tier 1: Is the label in ALWAYS_GOOD? → return 0 immediately, no area check needed.
        water_drop with area 9000px is still GOOD.

Tier 2: Is the label in CONDITIONAL_THRESHOLDS?
        → If area_px > threshold: return 1 (FAULTY)
        → If area_px ≤ threshold: return 0 (GOOD)
        → If area_px is missing: return 1 (conservative — assume worst case)
        Example: chip with area=150 → 150 ≤ 200 threshold → GOOD (0)
                 chip with area=250 → 250 > 200 threshold → FAULTY (1)

Tier 3: Is the label in ALWAYS_FAULTY? → return 1 immediately.
        glass_shard with area=0 is still FAULTY.

If none of the above match: return 1 (conservative unknown default).

**Parameters:**
- `label` — raw string label from CSV (case-insensitive, any separator format)
- `area_px` — defect area in pixels. Required for conditional labels. Pass None for unconditional ones.

**Returns:** Integer 0 (GOOD) or 1 (FAULTY).

---

### `resolve_labels_dataframe(df, label_col, area_col)`

```python
def resolve_labels_dataframe(df, label_col='label', area_col='area_px') -> pd.DataFrame:
```

**What it does:** Applies `resolve_single_label` to every row of the DataFrame
and adds a new column called 'binary_label'.

**Why it's needed:** Rather than resolving labels one at a time, this processes
the entire dataset at once and logs a summary of the class distribution.

**How it works:** Loops over every row, calls resolve_single_label for each,
stores results in a new column. Then counts how many 0s and 1s were produced
and prints the class balance.

**Parameters:**
- `df` — your annotation DataFrame (from pd.read_csv)
- `label_col` — name of the label column (default 'label')
- `area_col` — name of the area column (default 'area_px')

**Returns:** Copy of the input DataFrame with a new 'binary_label' column added.

---

## SECTION 3 — ROI Extraction Functions

ROI = Region of Interest. The raw camera image shows the bottle base surrounded
by the conveyor belt and background. The ROI extractor finds just the circular
bottle base and discards everything else.

### `detect_bottle_circle(gray_blurred, cfg, img_h, img_w)`

**What it does:** Finds the circular bottle base in a grayscale image using
the Hough Circle Transform algorithm.

**How Hough Circles work conceptually:**
Imagine drawing a circle of known radius centered at every pixel in the image.
For each circle, count how many edge pixels lie on it. The centre pixel whose
circle matches the most edge pixels is declared the circle's centre.
This "voting" process happens in an internal accumulator array.
param2 is the minimum vote count to accept a detection.

**Parameters:**
- `gray_blurred` — grayscale version of the image, already median-blurred
- `cfg` — CFG dictionary (for all the hough_ parameters)
- `img_h, img_w` — height and width of the image, used for tie-breaking

**What happens when multiple circles are found:**
The algorithm picks the circle whose centre is closest to the image centre.
This works because the camera is fixed above the conveyor — the bottle is
always roughly centred in the frame.

**Returns:** (cx, cy, radius) as three integers — centre x, centre y, and radius in pixels.

---

### `apply_clahe(bgr, clip_limit, tile_grid)`

**What it does:** Enhances the local contrast of an image using CLAHE
(Contrast Limited Adaptive Histogram Equalisation).

**Why it matters for bottle inspection:**
Some defects like contamination_light (just 180px to be FAULTY) and
glass_imperfection are extremely subtle — barely visible brightness differences
from the clean glass background. CLAHE makes these subtle differences much
more visible by locally stretching the contrast in each small tile of the image.

**Why LAB colour space?**
LAB separates the image into:
- L channel: brightness/luminance only
- A channel: green-red colour axis
- B channel: blue-yellow colour axis

Applying CLAHE only to L means we enhance brightness contrast without
touching colour information. If we applied it to BGR directly, a blue-ish
contamination would also have its colour shifted, losing the colour signal.

**Parameters:**
- `bgr` — image as a BGR numpy array (as returned by cv2.imread)
- `clip_limit` — maximum contrast amplification. 2.0 is conservative.
- `tile_grid` — (rows, cols) for the grid of tiles. (8,8) = 64 tiles.

**Returns:** BGR numpy array with enhanced local contrast.

---

### `pad_to_square(img)`

**What it does:** Adds black pixels to the shorter axis of a rectangular image
to make it perfectly square.

**Why it's needed:** After cropping the bounding box around a circle, the crop
might be slightly taller than wide or vice versa (if the circle is near an edge
and gets clipped). The neural network expects a square input.

**Why black padding?**
The circular mask already fills everything outside the bottle circle with
black pixels. Black padding blends seamlessly — there's no artificial border.

**Parameters:**
- `img` — BGR numpy array of any rectangular shape

**Returns:** BGR numpy array where height == width.

---

### `extract_roi_from_array(bgr, cfg)`

**What it does:** The complete ROI pipeline for a single image.
Takes a raw camera image and returns a 224×224 PIL Image of just the bottle base.

**Full pipeline step by step:**

Step 1 — Detect circle: Converts to grayscale, applies median blur (size 7),
         runs Hough detection. Median blur (not Gaussian) is used because it
         preserves edges while removing salt-and-pepper noise — the type of
         noise industrial cameras produce.

Step 2 — Circular mask: Creates a black image of the same size, then draws
         a white filled circle at the detected position. Uses cv2.bitwise_and
         to keep only the pixels inside the circle — everything outside becomes black.

Step 3 — Bounding crop: Calculates the rectangle that tightly contains the circle
         plus the margin. Clips coordinates to image boundaries (max/min) to
         handle bottles near the image edge.

Step 4 — Square padding: Ensures the crop is square.

Step 5 — CLAHE: Enhances local contrast.

Step 6 — Resize: Scales to input_size×input_size using INTER_AREA interpolation.
         INTER_AREA is the best algorithm when making images smaller (downsampling)
         — it averages pixel values rather than dropping them, preserving fine details.

Step 7 — Format conversion: BGR (OpenCV native) → RGB (what the neural network expects)
         → PIL Image (what Albumentations augmentation expects).

**Parameters:**
- `bgr` — raw image as BGR numpy array
- `cfg` — CFG dictionary

**Returns:** PIL Image, RGB, 224×224 pixels.

---

### `extract_roi_from_path(image_path, cfg)`

**What it does:** Convenience wrapper. Loads an image from disk using cv2.imread,
then calls extract_roi_from_array.

**Why it's separate:** During the pre-processing step (save crops to disk),
images are loaded from file paths. But during live inference from a camera feed,
images arrive as arrays already. Having two separate functions handles both cases.

**Parameters:**
- `image_path` — full file path as a string
- `cfg` — CFG dictionary

**Returns:** PIL Image, RGB, 224×224 pixels.

---

### `preprocess_and_save_all(df, output_dir, cfg, image_col, n_jobs)`

**What it does:** Runs ROI extraction on every image in the DataFrame and saves
the cropped result to disk. Run this once before training.

**Why pre-processing matters:**
Without this, every image loads the full raw file and runs Hough detection
during each training epoch. With 35,000 images and 50 epochs that's 1.75 million
Hough detections — very slow. After pre-processing, each epoch loads a pre-cropped
224×224 image directly — much faster.

**ThreadPoolExecutor:** Uses multiple CPU threads to process multiple images
simultaneously. With n_jobs=4, four images are being cropped at the same time.

**Parameters:**
- `df` — DataFrame with image paths
- `output_dir` — where to save the cropped images
- `cfg` — CFG dictionary
- `image_col` — column name containing image paths (default 'image_path')
- `n_jobs` — number of parallel threads

**Returns:** Copy of df where image_col now points to the pre-cropped files.

---

## SECTION 4 — Augmentation Pipeline

Augmentation artificially increases the variety of training images by applying
random transformations. The model sees a different random version of each image
every epoch, making it more robust to real-world variation.

Critical rule: augmentation is ONLY applied during training.
Validation and test images get resize + normalise only — deterministic and reproducible.

### `build_train_transforms(cfg)`

**What it does:** Returns an Albumentations Compose pipeline — a chain of
transformations applied in sequence to each training image.

**Why Albumentations instead of torchvision?**
Albumentations operates on numpy uint8 arrays (the native OpenCV format),
avoiding a costly float32 conversion. It's 3–10× faster than PIL-based transforms
for the same operations.

**Each transform explained:**

`A.Resize(size, size)` — Ensures the image is exactly input_size×input_size.
Even though pre-cropped images are already this size, this is a safety step.

`A.Rotate(limit=180, p=1.0)` — Rotates by a random angle between -180° and +180°.
p=1.0 means this is ALWAYS applied (probability 100%).
Why 180°? Bottle bases are circular and rotationally symmetric — a glass shard
at 45° is just as faulty as one at 225°. Full rotation multiplies the dataset
diversity without distorting meaningful information.

`A.HorizontalFlip(p=0.5)` — Mirrors the image left-right with 50% probability.
Again valid because bottle bases have no inherent left-right orientation.

`A.VerticalFlip(p=0.5)` — Mirrors the image top-bottom with 50% probability.

`A.RandomResizedCrop(scale=(0.85, 1.0))` — Randomly crops between 85% and 100%
of the image area, then resizes back to input_size. Simulates slight variation
in how centred the bottle is under the camera.

`A.ColorJitter(brightness=0.10, contrast=0.10)` — Randomly adjusts brightness
and contrast. Simulates camera exposure drift throughout the production day
(lighting conditions change as the factory heats up, lens gets dust, etc.).
Deliberately kept conservative (0.10 not 0.20) to avoid making clean bottles
look like they have contamination.

`A.Sharpen(alpha=(0.1, 0.3), p=0.3)` — Randomly sharpens the image.
Helps the model learn texture-based features needed to detect scuffing and
glass_imperfection, which are subtle texture changes rather than obvious blobs.

`A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 1.0), p=0.25)` —
Randomly blurs the image slightly. Simulates camera defocus or vibration blur
from the production line. p=0.25 means applied to 25% of images.

`A.GaussNoise(var_limit=(3.0, 15.0), p=0.2)` — Adds random sensor noise.
Simulates the electronic noise floor of an industrial camera sensor.

`A.GridDistortion(distort_limit=0.10, p=0.15)` — Applies a subtle random
warp to the image grid. Simulates barrel/pincushion lens distortion.

`A.Normalize(mean=..., std=...)` — ALWAYS applied last. Converts pixel values
from [0, 255] to the normalised range expected by the pretrained EfficientNet.

`ToTensorV2()` — Converts the numpy array (H, W, C) = (224, 224, 3) to a
PyTorch tensor (C, H, W) = (3, 224, 224). Neural networks process channels first.

---

### `build_eval_transforms(cfg)`

**What it does:** Returns the minimal deterministic pipeline for val/test images.

Only two steps: Resize (safety) → Normalize → ToTensorV2.
No randomness. The same image always produces the exact same tensor.
This is essential for validation metrics to be stable and comparable epoch-to-epoch.

---

## SECTION 5 — Dataset and DataLoader

### `BottleDataset` (class)

**What it does:** Connects the DataFrame (a table of image paths and labels) to
PyTorch's data loading system. Every time the training loop asks for a batch of
images, it calls __getitem__ for each image in the batch.

**Parameters:**
- `df` — DataFrame with image paths and binary_label column
- `transform` — the augmentation pipeline (train or eval)
- `cfg` — CFG dictionary
- `use_roi` — if True, runs the full ROI extraction pipeline on each load.
              if False (recommended after pre-processing), loads the pre-cropped image directly.

**`__len__`:** Returns the total number of images. Required by PyTorch.

**`__getitem__(idx)`:** The core function called by the DataLoader.
1. Gets the row for image at index idx
2. Loads the image (either via ROI extraction or direct PIL open)
3. Converts to numpy array (required by Albumentations)
4. Applies the augmentation pipeline
5. Returns (image_tensor, label_tensor) — a tuple PyTorch can batch

---

### `compute_pos_weight(train_df)`

**What it does:** Calculates a weight for the positive (FAULTY) class.

**Formula:** pos_weight = n_GOOD / n_FAULTY

With your numbers: 14729 / 20613 = 0.714

**What this means:** Since FAULTY is the majority class (58.3%), pos_weight < 1
tells the loss function to weight FAULTY errors 0.714× (less) than GOOD errors.
This counteracts the natural tendency of the model to over-predict the majority class.

---

### `make_weighted_sampler(labels)`

**What it does:** Creates a sampler that draws training images with probabilities
inversely proportional to their class frequency, achieving a balanced 50/50 batch ratio.

**Why needed:** Even after computing pos_weight for the loss function, the model
still sees the raw class imbalance during training. A batch of 64 images drawn
randomly from a 58% FAULTY dataset will have ~37 FAULTY and ~27 GOOD — the model
optimises more for FAULTY simply because there are more examples of it.
The weighted sampler ensures approximately 32 GOOD and 32 FAULTY per batch.

**How it works:**
1. Counts how many images exist per class
2. Assigns each class a weight = 1 / count (inverse frequency)
3. Assigns each INDIVIDUAL sample the weight of its class
4. WeightedRandomSampler draws samples according to these weights with replacement

`replacement=True` means the same image can appear multiple times in one epoch.
This is fine — Albumentations will apply different random augmentations each time,
making each occurrence effectively a different image.

---

### `make_dataloaders(df, cfg, use_roi)`

**What it does:** Orchestrates the entire data preparation — splits the dataset,
builds all three Datasets, creates the weighted sampler, and returns three
DataLoaders ready for training.

**Stratified split:** The split preserves class ratios.
If 41.7% of your dataset is GOOD, then your train, val, and test sets are
each also approximately 41.7% GOOD. Without stratification, you might get a
test set that's 60% GOOD by random chance, making metrics misleading.

**DataLoader parameters:**
- `sampler=sampler` — uses the weighted sampler for training (overrides shuffle)
- `pin_memory=True` — pins loaded tensors in CPU RAM for faster GPU transfer
- `persistent_workers=True` — keeps worker processes alive between epochs,
  avoiding the overhead of restarting them each epoch
- `prefetch_factor=2` (implicit) — each worker pre-loads 2 batches ahead

**Returns:** train_loader, val_loader, test_loader, pos_weight

---

## SECTION 6 — Model Architecture

### `build_model(cfg)`

**What it does:** Constructs the complete neural network by combining a pretrained
backbone (EfficientNet-B0) with a custom classification head.

**EfficientNet-B0 backbone (the feature extractor):**
This is a deep CNN with 7 stages of "MBConv" blocks (Mobile Inverted Bottleneck
Convolutions). The key concept: early layers detect simple patterns (edges,
gradients), later layers detect complex patterns (shapes, textures, objects).

`timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')`

- `pretrained=True` — loads weights trained on 1.28 million ImageNet images.
  These weights already "know" about textures, edges, shapes. We adapt them.
- `num_classes=0` — removes the original ImageNet classification head (1000 classes).
  We're replacing it with our own binary head.
- `global_pool='avg'` — Global Average Pooling. After the last conv layer,
  instead of keeping a 7×7×1280 feature map, it averages each of the 1280
  channels across the 7×7 spatial grid → output is [batch_size, 1280].
  This makes the model translation-invariant (doesn't care where in the image
  the defect is, only that it exists somewhere).

**Classification head (the decision maker):**

```
[batch, 1280]                   ← backbone output
    → Dropout(0.4)              ← randomly zero 40% during training
    → Linear(1280 → 256)        ← compress features to 256 dimensions
    → BatchNorm1d(256)          ← normalise 256 activations (explained below)
    → ReLU()                    ← non-linearity
    → Dropout(0.2)              ← secondary regularisation
    → Linear(256 → 1)           ← single output logit
```

**What is BatchNorm1d?**
During training, the distribution of activations in the 256-neuron layer can
shift dramatically from batch to batch. BatchNorm normalises them to have
mean=0 and std=1, then scales and shifts with learnable parameters.
This stabilises training and allows higher learning rates.

**What is ReLU?**
Rectified Linear Unit: f(x) = max(0, x).
Without non-linearities between layers, the entire network collapses to a
single linear equation (no matter how many layers). ReLU introduces the
non-linearity needed to learn complex decision boundaries.

**Why no Sigmoid at the output during training?**
The last layer outputs a raw "logit" — an unbounded number that can be positive
or negative. During training, `BCEWithLogitsLoss` and `focal_loss` apply the
sigmoid internally using the numerically stable log-sum-exp trick.
At inference, we explicitly call `torch.sigmoid(logit)` to get P(FAULTY) in [0,1].

**Kaiming initialisation:**
```python
nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
```
Random weight initialisation specifically designed for ReLU networks.
Regular random initialisation can cause vanishing gradients (gradients become
so small that early layers stop learning). Kaiming scaling prevents this.

---

### `freeze_backbone(model)`

**What it does:** Sets `requires_grad = False` on all backbone parameters.

**What `requires_grad` means:**
PyTorch tracks gradients (how much each weight contributed to the error) for
every parameter where `requires_grad=True`. Setting it to False means:
1. No gradient is computed for those parameters during backpropagation
2. Those parameters are not updated by the optimizer
3. Backpropagation stops at the frozen layer boundary (faster, less memory)

**Why freeze during phase 1:**
The head starts with random weights. If you immediately backpropagate errors
from random head predictions through the entire backbone, you corrupt the
carefully learned ImageNet features before the head has any meaningful signal.
Freezing lets the head stabilise first.

---

### `unfreeze_backbone_last_n(model, n=3)`

**What it does:** Unfreezes the last `n` child modules of the backbone,
keeping earlier layers frozen.

**Why only the last N layers?**
CNN layers learn increasingly specific features as you go deeper:
- First layers: universal edges, gradients (same in all images)
- Middle layers: textures, shapes (somewhat task-specific)
- Last layers: complex patterns (very task-specific)

For bottle inspection, early features (edges) from ImageNet still apply.
But the late features (ImageNet object parts) need to be retrained for
glass defects, transparent surfaces, and contamination patterns.
Unfreezing only the last 3 blocks gives a good tradeoff.

---

## SECTION 7 — Focal Loss

### `focal_loss(logits, targets, gamma, alpha, reduction)`

**What it does:** Computes how wrong the model's predictions are, with extra
emphasis on hard-to-classify examples.

**Standard cross-entropy loss (baseline to understand):**
Loss = -log(P(correct class))
If the model says P(FAULTY)=0.9 for a FAULTY bottle: loss = -log(0.9) = 0.105 (small)
If the model says P(FAULTY)=0.1 for a FAULTY bottle: loss = -log(0.1) = 2.303 (large)
Good so far — wrong predictions get penalised more.

**The problem:** On a balanced batch, 90% of images might be "easy" cases
where the model is already 95%+ confident. These dominate the total loss,
and the model stops caring about the hard borderline cases.

**Focal loss solution:** Multiply each sample's loss by (1 - p_t)^gamma.
- For an easy correct prediction (p_t = 0.95): weight = (1-0.95)^2 = 0.0025 — nearly ignored
- For a hard uncertain prediction (p_t = 0.50): weight = (1-0.50)^2 = 0.25 — full attention
- For a missed prediction (p_t = 0.10): weight = (1-0.10)^2 = 0.81 — heavily penalised

**Parameters:**
- `logits` — raw model output tensor [batch_size] before sigmoid
- `targets` — ground truth labels tensor [batch_size] with values 0.0 or 1.0
- `gamma` — focusing parameter. 0 = standard BCE. 2 = original paper default.
- `alpha` — class weight. Higher = model cares more about FAULTY errors.
- `reduction` — 'mean' averages loss over the batch (standard), 'sum' adds them.

**Implementation note — `F.binary_cross_entropy_with_logits`:**
This function computes BCE from raw logits using the log-sum-exp trick:
loss = max(x,0) - x*y + log(1 + exp(-|x|))
This is mathematically equivalent to sigmoid + BCE but avoids overflow
when logits are very large or very small (e.g. ±100).

---

## SECTION 8 — Learning Rate Schedule

### `lr_lambda(epoch, cfg)`

**What it does:** Returns a multiplier for the learning rate at each epoch.
PyTorch's LambdaLR scheduler calls this function and multiplies the
base learning rate (lr_phase1) by the returned value.

**Three segments:**

Segment 1 — Linear warmup (epochs 0 to warmup_epochs):
Returns (epoch+1)/warmup_epochs, growing from near 0 to 1.0.
Prevents the randomly-initialised head from making giant gradient steps
on its first few batches.

Segment 2 — Flat phase 1 (warmup_epochs to phase1_epochs):
Returns 1.0 (full lr_phase1). Head training at constant rate.

Segment 3 — Cosine annealing for phase 2 (phase1_epochs to total_epochs):
Returns a cosine curve from (lr_phase2/lr_phase1) down to (min_lr/lr_phase1).
The cosine shape: starts fast, slows down gradually, then very slow at the end.
This prevents overshooting the minimum near the end of training.

**Why cosine over linear decay?**
Linear decay reduces LR at a constant rate. Cosine decay reduces it slowly at
first (when the model still needs to learn) and rapidly at the end (when it's
close to converged and needs fine adjustments). Better match to how learning
actually progresses.

---

### `build_optimizer_and_scheduler(model, cfg)`

**What it does:** Creates the AdamW optimizer and LambdaLR scheduler.

**AdamW optimizer:**
Adam (Adaptive Moment Estimation) maintains a separate learning rate for each
parameter, adapting based on how that parameter's gradient has behaved historically.
Parameters that rarely change get higher effective LR; noisy parameters get lower LR.

AdamW = Adam + decoupled weight decay.
Regular Adam applies weight decay incorrectly (conflated with gradient update).
AdamW separates them, giving cleaner regularisation. Now the standard for
fine-tuning pretrained models.

`filter(lambda p: p.requires_grad, model.parameters())` — only passes
unfrozen (trainable) parameters to the optimizer. Passing frozen parameters
would waste memory and compute tracking gradients for them.

---

## SECTION 9 — Training Engine

### `train_one_epoch(model, loader, optimizer, scaler, cfg, device)`

**What it does:** One complete pass through the training DataLoader.
For each batch: loads images, runs the forward pass, computes loss,
runs backpropagation, updates weights.

**Step by step for each batch:**

1. `images = images.to(device, non_blocking=True)` — moves tensor from CPU RAM
   to GPU VRAM. `non_blocking=True` allows async transfer — the CPU can start
   preparing the next batch while the GPU is still processing the current one.

2. `optimizer.zero_grad(set_to_none=True)` — clears gradients from the previous
   batch. Gradients accumulate by default in PyTorch — without zeroing, batch N+1
   would contain gradients from batch N. `set_to_none=True` is faster than
   filling with zeros.

3. `with torch.cuda.amp.autocast()` — enables AMP. Operations like matrix
   multiplications run in fp16. Operations sensitive to precision (like loss
   computation) stay in fp32. PyTorch manages this automatically.

4. `logits = model(images).squeeze(1)` — the forward pass. Images go through
   EfficientNet backbone + head → [batch, 1] logits. `.squeeze(1)` removes
   the second dimension → [batch] tensor.

5. `loss = focal_loss(logits, labels, ...)` — measures how wrong the predictions are.

6. `scaler.scale(loss).backward()` — the backward pass (backpropagation).
   Computes how much each parameter contributed to the loss (the gradient).
   `scaler.scale` multiplies the loss by a large number before backward to
   prevent fp16 underflow (gradients too small to represent in fp16).

7. `scaler.unscale_(optimizer)` — divides gradients back by the scale factor
   before clipping. Clipping must happen on actual-magnitude gradients.

8. `clip_grad_norm_(model.parameters(), cfg['grad_clip'])` — if the total
   gradient norm exceeds grad_clip (1.0), all gradients are scaled down
   proportionally. Prevents catastrophic weight updates from bad batches.

9. `scaler.step(optimizer)` — updates all trainable parameters:
   parameter = parameter - learning_rate × gradient (simplified)

10. `scaler.update()` — adjusts the AMP scale factor for next batch.
    If gradients were valid: possibly increase scale.
    If gradients had inf/nan: decrease scale and skip the update.

**Returns:** Average loss over the epoch (for plotting).

---

### `evaluate(model, loader, device, cfg, threshold)`

**What it does:** Runs the model on an entire DataLoader without updating weights.
Collects all predictions and computes metrics.

`@torch.no_grad()` — decorator that disables gradient computation entirely.
During evaluation we don't need gradients (we're not training), so this saves
~40% of memory and ~20% speed compared to having them enabled.

`model.eval()` — switches BatchNorm and Dropout to evaluation mode:
- BatchNorm uses running statistics instead of batch statistics
- Dropout is turned off (all neurons active)
This is critical — training mode vs eval mode can change outputs significantly.

**Parameters:**
- `threshold` — the decision boundary. probs >= threshold → predicted FAULTY.

**Returns:** Dictionary with keys:
- `loss` — focal loss value
- `f1`, `precision`, `recall`, `accuracy` — standard classification metrics
- `roc_auc` — area under ROC curve (ranking quality, threshold-independent)
- `pr_auc` — area under precision-recall curve (better for imbalanced data)
- `probs` — numpy array of P(FAULTY) for every image (needed for calibration)
- `labels` — numpy array of true labels (needed for calibration)

---

### `train(model, train_loader, val_loader, cfg, device)`

**What it does:** The complete training loop — calls train_one_epoch and evaluate
in alternation, handles phase switching, checkpointing, and early stopping.

**Phase switching logic:**
```python
if epoch == cfg['phase1_epochs'] and current_phase == 1:
    unfreeze_backbone_last_n(model, n=3)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
```
At the phase 1 → phase 2 boundary, the backbone is partially unfrozen.
A new optimizer is created to include the newly unfrozen parameters.
A new scheduler starts fresh cosine annealing for phase 2.

**Checkpoint saving:**
```python
if val_f1 > best_f1 + cfg['early_stop_min_delta']:
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'val_f1': best_f1,
        'cfg': cfg,
    }, cfg['checkpoint_path'])
```
`model.state_dict()` is a dictionary of all model weight tensors.
This is what you reload later to restore the exact model state.
The epoch number and val_f1 are saved alongside for reference.

**Early stopping:**
`no_improve` counts consecutive epochs without meaningful F1 improvement.
Once it exceeds `early_stop_patience`, training stops. This saves compute
and prevents overfitting after the model has converged.

**Returns:** `history` dict with per-epoch train_loss, val_loss, val_f1, etc.
Used for the training history plot.

---

## SECTION 10 — Hyperparameter Tuning

### `optuna_objective(trial, train_loader, val_loader, base_cfg, n_epochs)`

**What it does:** Defines what one Optuna trial does — samples hyperparameters,
trains a model for n_epochs, and returns a score for Optuna to optimise.

**How Optuna works (TPE sampler):**
Trial 1: Random sample from search space
Trial 2: Random sample
...
Trial 5: Start building a probabilistic model:
  - Which hyperparameter values produced high F1? (the "good" distribution)
  - Which produced low F1? (the "bad" distribution)
  - Sample next trial from the good distribution (Expected Improvement)

This is smarter than random search (which doesn't learn from previous trials)
and faster than grid search (which tries all combinations).

**`trial.report(f1_val, epoch)` + `trial.should_prune()`:**
Reports intermediate results to Optuna after each epoch.
If this trial is performing significantly below the median of completed trials
at the same epoch, `should_prune()` returns True and we raise TrialPruned —
stopping the trial early to avoid wasting compute on a clearly bad configuration.

**The safety constraint:**
```python
if recall < cfg['min_recall_faulty']:
    return 0.0
```
Returns zero score for any configuration that misses too many FAULTY bottles,
regardless of how high its F1 might be. This prevents Optuna from finding
configurations that improve precision at the cost of safety.

**Parameters:**
- `trial` — Optuna trial object (provides suggest_* methods for sampling)
- `train_loader, val_loader` — data (already split)
- `base_cfg` — starting config (deep-copied so changes don't affect original)
- `n_epochs` — short budget (12 epochs) — enough to rank configurations, not full training

---

### `run_hyperparameter_search(train_loader, val_loader, cfg)`

**What it does:** Creates an Optuna study, runs n_trials objective calls,
and returns the best configuration found.

**Median pruner:** `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`
After 5 completed trials, starts pruning new trials whose intermediate score
falls below the median of completed trials at the same epoch step.
n_warmup_steps=3 means the first 3 epochs of each trial are never pruned
(too early to judge).

**Returns:** (best_cfg, study)
- `best_cfg` — copy of CFG with best hyperparameters applied
- `study` — full Optuna study object (can inspect all trials, plot importance, etc.)

---

## SECTION 11 — Threshold Calibration

### `compute_metrics_at_threshold(labels, probs, threshold)`

**What it does:** Given predicted probabilities and true labels, applies a
specific threshold and computes all classification metrics.

**The confusion matrix:**
```
              Predicted GOOD    Predicted FAULTY
True GOOD         TN                 FP
True FAULTY       FN                 TP
```
- TP (True Positive): Correctly identified FAULTY bottles
- TN (True Negative): Correctly identified GOOD bottles
- FP (False Positive): GOOD bottles wrongly called FAULTY (waste)
- FN (False Negative): FAULTY bottles wrongly called GOOD (safety risk!)

**Metrics computed:**
- precision_faulty = TP / (TP + FP) — of all bottles called FAULTY, how many are really?
- recall_faulty = TP / (TP + FN) — of all truly FAULTY bottles, how many did we catch?
- f1_faulty = 2 × precision × recall / (precision + recall) — harmonic mean
- roc_auc — area under ROC curve. 0.5 = random, 1.0 = perfect.
- pr_auc — area under precision-recall curve. More informative for imbalanced data.

---

### `calibrate_threshold(val_probs, val_labels, min_recall)`

**What it does:** Searches for the best decision threshold on the validation set.

**How precision_recall_curve works:**
sklearn computes precision and recall at every possible threshold in one pass.
Returns arrays of equal length — precision[i], recall[i], threshold[i].

**The search:**
For each candidate threshold:
1. Check: does recall_faulty >= min_recall? If not, skip entirely (safety gate)
2. Compute F1 at this threshold
3. If F1 > best seen so far, save this threshold

**Why calibrate on validation, not test?**
The threshold is a hyperparameter. If we optimised it on the test set, we'd
be fitting to the test set — reporting falsely high performance.
Calibrating on validation and reporting test metrics is honest.

**Why is 0.5 almost never optimal for imbalanced data?**
The model's raw output is calibrated relative to the training distribution.
With 58% FAULTY in training, the model's "natural" midpoint is biased toward
FAULTY. The calibrator compensates by finding the threshold where the
distributions actually cross, which may be much lower or higher than 0.5.

---

## SECTION 12 — Evaluation and Visualisation

### `print_evaluation_report(metrics, split)`

Prints a formatted summary of all metrics to stdout.
The "✓ PASSED / ✗ BELOW TARGET" line checks whether F1 ≥ 0.98.

---

### `plot_training_history(history, output_dir)`

Three side-by-side charts:
1. **Loss curve** — train and val loss per epoch. Should both decrease. If train loss
   decreases but val loss increases: overfitting. If both plateau: converged.
2. **Val F1** — the primary metric trend. Should approach 0.98 line.
3. **Learning rate** — shows the warmup, flat phase 1, and cosine decay clearly.

---

### `plot_evaluation_charts(probs, labels, threshold, output_dir)`

Four charts saved as a single image:

1. **Confusion matrix** — the four cells (TP, TN, FP, FN) as a heatmap.
   Large FP number = precision problem. Large FN number = recall problem.

2. **Precision-Recall curve** — plots precision vs recall at every possible threshold.
   The dashed vertical line shows where your calibrated threshold sits.
   Ideally the curve hugs the top-right corner. AP (area under curve) close to 1.0
   means the model ranks FAULTY above GOOD almost perfectly.

3. **ROC curve** — plots True Positive Rate vs False Positive Rate.
   AUC close to 1.0 means excellent separation. The diagonal is random chance.

4. **Confidence histogram** — most diagnostic chart.
   Shows P(FAULTY) distribution separately for GOOD bottles (blue) and FAULTY (red).
   Ideal: GOOD bottles concentrated near 0, FAULTY near 1, minimal overlap.
   Your current chart shows heavy overlap at 0.3–0.7 — this is what the training
   fixes are designed to eliminate.

---

### `export_predictions(test_loader, probs, preds, labels, threshold, output_dir)`

Saves a CSV with one row per test image containing every relevant piece of information:
- prob_faulty: the exact P(FAULTY) the model predicted
- pred_class: 'GOOD' or 'FAULTY' based on the threshold
- true_class: the actual label
- correct: True/False
- threshold_used: the calibrated threshold that was applied

This CSV is your primary debugging tool. Sort by `prob_faulty` descending and
look at the top false positives — the GOOD bottles the model was most certain
were FAULTY. Those images reveal exactly what visual feature is confusing the model.

---

## SECTION 13 — Single Image Inference

### `predict_single_image(image_path, model, threshold, cfg, device, show)`

**What it does:** Runs the complete pipeline on one new image and returns
a prediction dictionary.

**Steps:**
1. ROI extraction (full Hough + CLAHE + resize pipeline)
2. Build eval transform pipeline (no augmentation)
3. `.unsqueeze(0)` — adds a batch dimension: [3, 224, 224] → [1, 3, 224, 224].
   The model always expects a batch, even when predicting one image.
4. Forward pass under `torch.no_grad()` and autocast
5. Apply sigmoid to get P(FAULTY) in [0,1]
6. Compare to threshold → binary decision

**`confident` flag:**
If the predicted probability is more than 0.2 away from the threshold,
the prediction is flagged as "confident". If it's within 0.2 of the threshold,
it's flagged as "Borderline" — a human should review this bottle.

---

## Quick Reference: What to change when things go wrong

| Problem | Parameter to change | Direction |
|---|---|---|
| Too many GOOD bottles rejected (high FP) | `focal_alpha` | Decrease toward 0.3 |
| Too many FAULTY bottles missed (high FN) | `focal_alpha` | Increase toward 0.6 |
| Model not learning (flat loss) | `lr_phase1` | Increase 2-5× |
| Loss explodes / NaN | `grad_clip` | Decrease to 0.5 |
| Overfitting (val loss rises) | `dropout1`, `dropout2` | Increase by 0.1 |
| Underfitting (both losses plateau high) | `total_epochs` | Increase; try full unfreeze |
| GPU out of memory | `batch_size` | Halve it |
| Training too slow | `num_workers` | Increase to 8 |
| F1 good on val, bad on test | Check stratified split | Verify class ratios match |
