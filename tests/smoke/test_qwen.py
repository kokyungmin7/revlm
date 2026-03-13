from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_qwen_align_produces_bgr_image(sample_image_bgr: np.ndarray) -> None:
    from src.preprocessing.align_angle import align_to_angle, load_qwen_angle_model

    pipe = load_qwen_angle_model()
    result = align_to_angle(pipe, sample_image_bgr, current_angle=90, target_angle=0)

    assert result.ndim == 3 and result.shape[2] == 3, (
        f"Expected (H, W, 3) output, got shape {result.shape}"
    )

    print(f"\n[smoke_qwen] output shape: {result.shape}")
    print(f"  input shape:  {sample_image_bgr.shape}")
    print(f"  → saved: experiments/smoke_qwen_compare.jpg")

    Path("experiments").mkdir(exist_ok=True)

    h_orig, w_orig = sample_image_bgr.shape[:2]
    h_res, w_res = result.shape[:2]
    target_h = max(h_orig, h_res)

    def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == height:
            return img
        pad = np.zeros((height - h, w, 3), dtype=img.dtype)
        return np.vstack([img, pad])

    compare = np.hstack([pad_to_height(sample_image_bgr, target_h), pad_to_height(result, target_h)])
    cv2.imwrite("experiments/smoke_qwen_compare.jpg", compare)
