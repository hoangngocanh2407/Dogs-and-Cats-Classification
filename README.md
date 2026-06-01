# Dogs and Cats Classification

This project trains a classical computer-vision classifier that distinguishes cats from dogs using COCO bounding-box annotations. It extracts HOG features from each annotated animal crop, trains an SVM classifier, saves the trained pipeline, and provides a small GUI for drawing a bounding box on a new image.

## Project structure

```text
.
├── DAPproject.py          # CLI for training and GUI inference
├── DaC.v3i.coco.zip      # Dataset archive
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

Generated files are intentionally ignored by Git:

- `DaC.v3i.coco/` after extracting the dataset
- `artifacts/` after training the model
- Python cache files and virtual environments

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

> Note: the GUI command also needs Tkinter. On Linux, install the OS package if it is missing, for example `sudo apt install python3-tk`.

## Train the model

The training command automatically extracts `DaC.v3i.coco.zip` if `DaC.v3i.coco/` is not already present:

```bash
python DAPproject.py train --dataset-zip DaC.v3i.coco.zip
```

By default, the command reads `DaC.v3i.coco/train/_annotations.coco.json`, extracts HOG features from each annotation bounding box, trains a scaler + SVM pipeline, and saves outputs to `artifacts/`:

- `model.joblib`
- `metadata.json`
- `classification_report.json`
- `confusion_matrix.json`

Useful options:

```bash
python DAPproject.py train --help
python DAPproject.py train --output-dir artifacts --test-size 0.2 --kernel linear --c 1.0
```

## Run the GUI

After training, start the interactive GUI:

```bash
python DAPproject.py gui
```

Click **Select Image**, choose an image, draw a bounding box around the animal, and press `q` in the OpenCV image window to close it. You can also preselect an image path:

```bash
python DAPproject.py gui --image path/to/image.jpg
```

## Current model approach

- Dataset format: COCO annotations
- Feature extractor: HOG
- Classifier: scikit-learn SVM
- Preprocessing: crop annotation bounding box, convert to grayscale, resize to `64x128`, extract HOG features, standardize features

## Recommended future improvements

- Add a dedicated validation/test split instead of splitting annotations from the training folder.
- Save a visual confusion-matrix plot in addition to JSON reports.
- Add automated tests for bbox normalization and feature extraction.
- Try transfer learning with a CNN such as MobileNet, EfficientNet, or ResNet for better accuracy.
- Train an object detector if the final goal is automatic detection without manually drawing a bounding box.
