"""Tests for src/preprocessing/detect.py."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.preprocessing.detect import DetectResult, detect_and_crop


def _make_image(h: int = 100, w: int = 60) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detect_disabled_returns_full_image() -> None:
    """enabled=False should return the original image as a single crop with no boxes."""
    image = _make_image()
    result = detect_and_crop(image, model=None, enabled=False)

    assert isinstance(result, DetectResult)
    assert len(result.crops) == 1
    assert np.array_equal(result.crops[0], image)
    assert result.boxes == []
    assert np.array_equal(result.source, image)


def test_detect_with_mock_yolo_returns_crop() -> None:
    """YOLO mock returning one person bbox should produce one crop."""
    image = _make_image(h=100, w=80)

    # Build a mock box: xyxy = [[10, 10, 50, 90]]
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [10.0, 10.0, 50.0, 90.0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model = MagicMock(return_value=[mock_result])

    result = detect_and_crop(image, model=mock_model, enabled=True, conf=0.5)

    assert len(result.crops) == 1
    assert len(result.boxes) == 1
    assert result.boxes[0] == (10, 10, 50, 90)
    expected_crop = image[10:90, 10:50]
    assert np.array_equal(result.crops[0], expected_crop)


def test_detect_no_person_returns_empty() -> None:
    """YOLO returning no detections should produce empty crops and boxes."""
    image = _make_image()

    mock_result = MagicMock()
    mock_result.boxes = []

    mock_model = MagicMock(return_value=[mock_result])

    result = detect_and_crop(image, model=mock_model, enabled=True)

    assert result.crops == []
    assert result.boxes == []
