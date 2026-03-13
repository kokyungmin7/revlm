"""Step 1: YOLO-based person detection and crop.

Supports two modes:
- enabled=True : Run YOLO inference and crop each detected person bbox.
- enabled=False: Return the full image as a single crop (pre-cropped dataset mode).
"""
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectResult:
    """Result of detect_and_crop.

    Attributes:
        crops: Person crop images in BGR format.
        boxes: Bounding boxes as (x1, y1, x2, y2) tuples.
        source: Original input image.
    """

    crops: list[np.ndarray]
    boxes: list[tuple[int, int, int, int]]
    source: np.ndarray


def detect_and_crop(
    image: np.ndarray,
    model: Any,
    enabled: bool = True,
    conf: float = 0.5,
) -> DetectResult:
    """Detect persons and return crops.

    Args:
        image: Input BGR image (H x W x 3).
        model: Loaded YOLO model instance.
        enabled: If False, skip detection and return the full image as a single crop.
        conf: Confidence threshold for YOLO detection.

    Returns:
        DetectResult with crops and bounding boxes.
    """
    if not enabled:
        return DetectResult(crops=[image], boxes=[], source=image)

    results = model(image, conf=conf, classes=[0])  # class 0 = person

    crops: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            boxes.append((x1, y1, x2, y2))

    return DetectResult(crops=crops, boxes=boxes, source=image)
