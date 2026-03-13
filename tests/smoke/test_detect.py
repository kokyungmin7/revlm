import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

YOLO_PATH = Path("models/detect/yolo26n-pose.pt")
if not YOLO_PATH.exists():
    pytest.skip("YOLO weights not found", allow_module_level=True)


def test_detect_real_model(sample_image_bgr: np.ndarray) -> None:
    from ultralytics import YOLO

    from src.preprocessing.detect import detect_and_crop

    model = YOLO(str(YOLO_PATH))
    result = detect_and_crop(sample_image_bgr, model, enabled=True, conf=0.5)

    assert isinstance(result.crops, list)

    print(f"\n[smoke_detect] crops detected: {len(result.crops)}")
    for i, box in enumerate(result.boxes):
        print(f"  crop #{i}: box={box}")
    print(f"  → saved: experiments/smoke_detect_out.jpg, smoke_detect_crop_*.jpg")

    import cv2

    Path("experiments").mkdir(exist_ok=True)

    annotated = sample_image_bgr.copy()
    for i, (crop, box) in enumerate(zip(result.crops, result.boxes)):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"#{i}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(f"experiments/smoke_detect_crop_{i}.jpg", crop)

    cv2.imwrite("experiments/smoke_detect_out.jpg", annotated)


def test_detect_crops_have_valid_shape(sample_image_bgr: np.ndarray) -> None:
    from ultralytics import YOLO

    from src.preprocessing.detect import detect_and_crop

    model = YOLO(str(YOLO_PATH))
    result = detect_and_crop(sample_image_bgr, model, enabled=True, conf=0.5)

    print(f"\n[smoke_detect] shape check: {len(result.crops)} crops")
    for i, crop in enumerate(result.crops):
        print(f"  crop #{i}: shape={crop.shape}")

    if len(result.crops) == 0:
        warnings.warn("No persons detected in sample image. Try a different image.")
        return

    for crop in result.crops:
        assert crop.ndim == 3 and crop.shape[2] == 3, (
            f"Expected (H, W, 3) crop, got shape {crop.shape}"
        )
