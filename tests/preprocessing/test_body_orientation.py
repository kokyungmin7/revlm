"""Tests for src/preprocessing/body_orientation.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.preprocessing.body_orientation import predict_orientation


def _make_bgr_crop(h: int = 256, w: int = 192) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mock_model_with_argmax(argmax_index: int) -> MagicMock:
    """Return a mock model whose hoe_output argmax equals argmax_index."""
    hoe_logits = torch.zeros(72)
    hoe_logits[argmax_index] = 1.0

    mock_model = MagicMock()
    mock_model.return_value = (None, hoe_logits.unsqueeze(0))
    return mock_model


def test_predict_argmax_to_angle() -> None:
    """argmax index 9 should map to angle 45 (9 * 5 = 45)."""
    model = _mock_model_with_argmax(9)
    image = _make_bgr_crop()

    angle = predict_orientation(model, image, device="cpu")

    assert angle == 45


def test_predict_argmax_zero() -> None:
    """argmax index 0 should map to angle 0."""
    model = _mock_model_with_argmax(0)
    image = _make_bgr_crop()

    angle = predict_orientation(model, image, device="cpu")

    assert angle == 0


def test_predict_argmax_max_class() -> None:
    """argmax index 71 should map to angle 355 (71 * 5 = 355)."""
    model = _mock_model_with_argmax(71)
    image = _make_bgr_crop()

    angle = predict_orientation(model, image, device="cpu")

    assert angle == 355
