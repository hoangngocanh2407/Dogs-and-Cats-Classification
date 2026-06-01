"""Train and run a HOG + SVM dogs-vs-cats classifier.

This module keeps the original project approach (COCO bounding boxes -> HOG
features -> linear SVM) but makes it portable and easier to run from any clone.

Examples:
    python DAPproject.py train --dataset-zip DaC.v3i.coco.zip
    python DAPproject.py gui --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("DaC.v3i.coco")
DEFAULT_DATASET_ZIP = Path("DaC.v3i.coco.zip")
DEFAULT_OUTPUT_DIR = Path("artifacts")
ANNOTATION_FILE = "_annotations.coco.json"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HOG_IMAGE_SIZE = (64, 128)  # width, height expected by the HOG extractor
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingMetadata:
    """Metadata saved next to the model so inference is reproducible."""

    dataset_dir: str
    split: str
    hog_image_size: tuple[int, int]
    hog_orientations: int
    hog_pixels_per_cell: tuple[int, int]
    hog_cells_per_block: tuple[int, int]
    random_state: int
    category_id_to_name: dict[int, str]
    samples: int
    train_samples: int
    test_samples: int
    accuracy: float


def import_ml_dependencies() -> dict[str, Any]:
    """Import optional ML dependencies only when a command needs them."""

    try:
        import cv2
        import joblib
        import numpy as np
        from PIL import Image
        from skimage.feature import hog
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - environment dependent
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    return locals()


def ensure_dataset(dataset_dir: Path, dataset_zip: Path | None) -> Path:
    """Return an extracted dataset directory, extracting the zip when needed."""

    dataset_dir = dataset_dir.expanduser().resolve()
    if (dataset_dir / "train" / ANNOTATION_FILE).exists():
        return dataset_dir

    if dataset_zip is None:
        raise FileNotFoundError(
            f"Could not find {dataset_dir / 'train' / ANNOTATION_FILE}. "
            "Pass --dataset-zip or extract the dataset first."
        )

    dataset_zip = dataset_zip.expanduser().resolve()
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {dataset_zip}")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {dataset_zip} to {dataset_dir.parent} ...")
    with zipfile.ZipFile(dataset_zip) as archive:
        archive.extractall(dataset_dir.parent)

    if not (dataset_dir / "train" / ANNOTATION_FILE).exists():
        raise FileNotFoundError(
            f"Extraction completed, but {dataset_dir / 'train' / ANNOTATION_FILE} was not found."
        )
    return dataset_dir


def load_coco_annotations(split_dir: Path) -> tuple[dict[str, Any], dict[int, str]]:
    """Load COCO annotations and category labels for a split directory."""

    annotations_path = split_dir / ANNOTATION_FILE
    with annotations_path.open("r", encoding="utf-8") as annotation_file:
        coco = json.load(annotation_file)

    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    if not category_id_to_name:
        raise ValueError(f"No categories found in {annotations_path}")
    return coco, category_id_to_name


def normalize_bbox_xywh(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    """Convert and clamp a COCO [x, y, width, height] bbox to xyxy."""

    x, y, width, height = bbox
    x1 = max(0, min(image_width, int(round(x))))
    y1 = max(0, min(image_height, int(round(y))))
    x2 = max(0, min(image_width, int(round(x + width))))
    y2 = max(0, min(image_height, int(round(y + height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def normalize_bbox_xyxy(
    x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    """Normalize a user-drawn bbox, allowing drags in any direction."""

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0, min(image_width, left))
    right = max(0, min(image_width, right))
    top = max(0, min(image_height, top))
    bottom = max(0, min(image_height, bottom))
    if right - left < 8 or bottom - top < 8:
        return None
    return left, top, right, bottom


def extract_hog_features_from_array(image: Any, bbox_xyxy: tuple[int, int, int, int], deps: dict[str, Any]) -> Any:
    """Extract one HOG feature vector from an image array and xyxy bbox."""

    cv2 = deps["cv2"]
    hog = deps["hog"]

    x1, y1, x2, y2 = bbox_xyxy
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Cannot extract features from an empty bounding box.")

    if roi.ndim == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray_roi = roi

    resized_roi = cv2.resize(gray_roi, HOG_IMAGE_SIZE)
    return hog(
        resized_roi,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )


def build_feature_matrix(split_dir: Path, deps: dict[str, Any]) -> tuple[Any, Any, dict[int, str]]:
    """Build feature and label arrays from COCO annotations."""

    np = deps["np"]
    Image = deps["Image"]

    coco, category_id_to_name = load_coco_annotations(split_dir)
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    features: list[Any] = []
    labels: list[int] = []
    skipped = 0

    for annotation in coco.get("annotations", []):
        image_info = images_by_id.get(int(annotation["image_id"]))
        if image_info is None:
            skipped += 1
            continue

        image_path = split_dir / image_info["file_name"]
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES or not image_path.exists():
            skipped += 1
            continue

        with Image.open(image_path) as pil_image:
            rgb_image = np.array(pil_image.convert("RGB"))

        image_height, image_width = rgb_image.shape[:2]
        bbox = normalize_bbox_xywh(annotation["bbox"], image_width, image_height)
        if bbox is None:
            skipped += 1
            continue

        features.append(extract_hog_features_from_array(rgb_image, bbox, deps))
        labels.append(int(annotation["category_id"]))

    if not features:
        raise ValueError(f"No usable training samples found in {split_dir}")

    if skipped:
        print(f"Skipped {skipped} annotations with missing images or invalid boxes.")

    return np.asarray(features), np.asarray(labels), category_id_to_name


def train(args: argparse.Namespace) -> None:
    """Train and evaluate the classifier."""

    deps = import_ml_dependencies()
    joblib = deps["joblib"]
    train_test_split = deps["train_test_split"]
    Pipeline = deps["Pipeline"]
    StandardScaler = deps["StandardScaler"]
    SVC = deps["SVC"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]

    dataset_dir = ensure_dataset(args.dataset_dir, args.dataset_zip)
    split_dir = dataset_dir / args.split
    features, labels, category_id_to_name = build_feature_matrix(split_dir, deps)

    stratify = labels if len(set(labels.tolist())) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=args.kernel, C=args.c, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(model.score(x_test, y_test))
    report = classification_report(
        y_test,
        predictions,
        target_names=[category_id_to_name.get(int(label), str(label)) for label in sorted(set(labels.tolist()))],
        labels=sorted(set(labels.tolist())),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=sorted(set(labels.tolist()))).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib")

    metadata = TrainingMetadata(
        dataset_dir=str(dataset_dir),
        split=args.split,
        hog_image_size=HOG_IMAGE_SIZE,
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2),
        random_state=RANDOM_STATE,
        category_id_to_name=category_id_to_name,
        samples=int(len(labels)),
        train_samples=int(len(y_train)),
        test_samples=int(len(y_test)),
        accuracy=accuracy,
    )
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(asdict(metadata), metadata_file, indent=2)
    with (args.output_dir / "classification_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    with (args.output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as matrix_file:
        json.dump(matrix, matrix_file, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved model and reports to {args.output_dir}")


def load_model_bundle(model_dir: Path, deps: dict[str, Any]) -> tuple[Any, dict[int, str]]:
    """Load the trained pipeline and category labels."""

    joblib = deps["joblib"]
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected {model_path} and {metadata_path}. Run `python DAPproject.py train` first."
        )
    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    labels = {int(key): value for key, value in metadata["category_id_to_name"].items()}
    return model, labels


def predict_bbox(model: Any, labels: dict[int, str], image: Any, bbox: tuple[int, int, int, int], deps: dict[str, Any]) -> str:
    """Predict a label name for one image bbox."""

    features = extract_hog_features_from_array(image, bbox, deps)
    prediction = int(model.predict([features])[0])
    return labels.get(prediction, str(prediction))


def run_gui(args: argparse.Namespace) -> None:
    """Open an interactive OpenCV window for drawing prediction boxes."""

    deps = import_ml_dependencies()
    cv2 = deps["cv2"]

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Tkinter is required for the GUI command.") from exc

    model, labels = load_model_bundle(args.model_dir, deps)

    root = tk.Tk()
    root.title("Dogs and Cats Classifier")

    def select_image() -> None:
        file_path = args.image or filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        bgr_image = cv2.imread(str(file_path))
        if bgr_image is None:
            print(f"Could not read image: {file_path}")
            return

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        display_image = bgr_image.copy()
        start_point: tuple[int, int] | None = None

        def draw_bbox(event: int, x: int, y: int, flags: int, _param: object) -> None:
            nonlocal display_image, start_point
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and start_point:
                display_image = bgr_image.copy()
                cv2.rectangle(display_image, start_point, (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP and start_point:
                bbox = normalize_bbox_xyxy(
                    start_point[0], start_point[1], x, y, rgb_image.shape[1], rgb_image.shape[0]
                )
                display_image = bgr_image.copy()
                if bbox is None:
                    print("Bounding box is too small or invalid.")
                    start_point = None
                    return

                label_name = predict_bbox(model, labels, rgb_image, bbox, deps)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_image,
                    f"Label: {label_name}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                start_point = None

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_bbox)
        while True:
            cv2.imshow("Image", display_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    button = tk.Button(root, text="Select Image", command=select_image)
    button.pack(padx=10, pady=10)
    root.mainloop()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Dogs and cats HOG + SVM classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the classifier")
    train_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--dataset-zip", type=Path, default=DEFAULT_DATASET_ZIP)
    train_parser.add_argument("--split", default="train")
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--kernel", default="linear")
    train_parser.add_argument("--c", type=float, default=1.0)
    train_parser.set_defaults(func=train)

    gui_parser = subparsers.add_parser("gui", help="Run the interactive prediction GUI")
    gui_parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gui_parser.add_argument("--image", type=Path, default=None)
    gui_parser.set_defaults(func=run_gui)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Program entry point."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.func(args)


if __name__ == "__main__":
    main()
