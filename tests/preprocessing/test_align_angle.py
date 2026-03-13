"""Tests for src/preprocessing/align_angle.py."""
from unittest.mock import MagicMock, call

import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.align_angle import align_to_angle


def _make_bgr_image(h: int = 64, w: int = 48) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mock_pipe(output_image: np.ndarray) -> MagicMock:
    """Return a mock pipeline whose result is output_image (RGB PIL)."""
    pil_out = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [pil_out]
    return mock_pipe


def test_align_same_angle_returns_original_without_pipe_call() -> None:
    """When |delta| < 5, return the original image and never call the pipeline."""
    image = _make_bgr_image()
    pipe = MagicMock()

    result = align_to_angle(pipe, image, current_angle=90, target_angle=90)

    pipe.assert_not_called()
    assert np.array_equal(result, image)


def test_align_nearly_same_angle_returns_original() -> None:
    """Delta of 4° is below threshold — pipeline must not be called."""
    image = _make_bgr_image()
    pipe = MagicMock()

    result = align_to_angle(pipe, image, current_angle=0, target_angle=4)

    pipe.assert_not_called()
    assert np.array_equal(result, image)


def test_prompt_contains_left_and_degrees() -> None:
    """current=90, target=0 → delta=-90 → prompt contains 'left' and '90'."""
    image = _make_bgr_image()
    output = _make_bgr_image()
    pipe = _mock_pipe(output)

    align_to_angle(pipe, image, current_angle=90, target_angle=0)

    prompt_used: str = pipe.call_args.kwargs.get("prompt") or pipe.call_args[1].get("prompt") or pipe.call_args[0][1]
    assert "left" in prompt_used.lower()
    assert "90" in prompt_used


def test_prompt_contains_right_and_degrees() -> None:
    """current=0, target=90 → delta=+90 → prompt contains 'right' and '90'."""
    image = _make_bgr_image()
    output = _make_bgr_image()
    pipe = _mock_pipe(output)

    align_to_angle(pipe, image, current_angle=0, target_angle=90)

    prompt_used: str = pipe.call_args.kwargs.get("prompt") or pipe.call_args[1].get("prompt") or pipe.call_args[0][1]
    assert "right" in prompt_used.lower()
    assert "90" in prompt_used


def test_align_returns_bgr_image() -> None:
    """Output of align_to_angle must be a valid BGR np.ndarray."""
    image = _make_bgr_image()
    output = _make_bgr_image()
    pipe = _mock_pipe(output)

    result = align_to_angle(pipe, image, current_angle=0, target_angle=180)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3
