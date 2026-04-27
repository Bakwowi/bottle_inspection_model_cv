# Bottle Inspection Computer Vision Project

## Overview

This project implements an automated bottle inspection system using Computer Vision and Deep Learning techniques. The system is designed to detect various defects in bottles during manufacturing quality control processes, utilizing YOLO (You Only Look Once) object detection models for real-time defect identification.

## Features

- **Defect Detection**: Identifies multiple types of bottle defects including:
  - Air bubbles
  - Chips and cracks
  - Contamination (light and dark)
  - Glass imperfections
  - Scuffing (regular and heavy)
  - Foreign objects
  - Liquid presence
  - Mold residues
  - Yeast residues

- **Image Preprocessing**: Advanced preprocessing pipeline including:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
  - Dynamic ROI cropping based on annotations
  - Annotation coordinate adjustment for cropped images

- **Quality Control Logic**: Intelligent decision-making system with:
  - Conditional fault thresholds (size-based rejection)
  - Always-faulty defect categories
  - Cumulative area analysis for defect clusters

## Project Structure

```
├── bottle_inspection.ipynb     # Main Jupyter notebook with complete pipeline
├── script.ipynb               # Additional scripts and experiments
├── image_data.yaml            # YOLO dataset configuration
├── yolo26n.pt                 # YOLOv8 nano model weights
├── readme.md                  # Project documentation
├── .gitignore                 # Git ignore rules
├── dataset/            # Main dataset directory
│   ├── train_images/          # Training images
│   ├── test_images/           # Test images
│   ├── final_train_images/    # Preprocessed training images
│   ├── train_annotations.json # COCO format annotations
│   ├── adjusted_annotations.json # Processed annotations
│   └── sample_submission.csv  # Competition submission format
└── models/                    # Trained model outputs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bottle-inspection-model_cv
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python numpy matplotlib tqdm
   ```

## Usage

### Data Preparation

1. **Preprocess Images**: Run the preprocessing pipeline to enhance images and adjust annotations:
   ```python
   from bottle_inspection import crop_image_and_adjust_annotations

   crop_image_and_adjust_annotations(
       json_path="dataset/train_annotations.json",
       img_dir="dataset/train_images",
       output_dir="dataset"
   )
   ```

2. **Convert Annotations**: Convert COCO format to YOLO format:
   ```python
   from ultralytics.data.converter import convert_coco

   convert_coco(
       labels_dir="dataset/annotations.json",
       save_dir="dataset",
       use_segments=False
   )
   ```

### Model Training

1. **Hyperparameter Evolution**:
   ```python
   from ultralytics import YOLO

   model = YOLO('yolo26n.pt')
   model.evolve(
       data='image_data.yaml',
       epochs=10,
       iterations=100,
       imgsz=640
   )
   ```

2. **Final Training**:
   ```python
   model.train(
       data='image_data.yaml',
       epochs=150,
       imgsz=640,
       batch=32,
       patience=25
   )
   ```

### Inference and Quality Control

```python
# Load trained model
model = YOLO('models/best.pt')

# Run inference
results = model.predict(source='path/to/test/image.jpg')

# Quality control decision
decision, reason = final_decision(results)
print(f"Decision: {decision} - {reason}")
```

## Configuration

### Quality Control Thresholds

The system uses configurable thresholds for defect classification:

```python
CONDITIONALLY_FAULTY = {
    "Air bubble": {"single": 500, "total": 1200},
    "Chip": {"single": 200, "total": 400},
    # ... additional thresholds
}

ALWAYS_FAULTY = [
    "Break / Crack", "Contamination dark", "Foreign object",
    # ... critical defects
]
```

### Model Configuration

Dataset configuration in `image_data.yaml`:
```yaml
path: ./dataset
train: train_images
val: test_images
names:
  0: Air bubble
  1: Break / Crack
  # ... class definitions
```

## Results and Evaluation

The model performance can be evaluated using:

```python
# Validation metrics
results = model.val(data='image_data.yaml', imgsz=640)

print(f"mAP@50: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
```

<!-- ## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request -->

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details. -->

## Acknowledgments

<!-- - Dataset provided by Krones AG for the bottle inspection challenge -->
- YOLOv8 implementation by Ultralytics
- OpenCV for computer vision operations