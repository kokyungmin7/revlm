"""Smoke tests for VLM-based ReID verification module.

Requires CUDA environment (Qwen3-VL-8B-Instruct, ~16GB bfloat16).
Run on GCP L4 or equivalent CUDA GPU.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

if not torch.cuda.is_available():
    pytest.skip("CUDA not available — skipping VLM verifier smoke tests", allow_module_level=True)


@pytest.fixture(scope="module")
def verifier():
    from src.models.vlm_verifier import load_vlm_verifier

    return load_vlm_verifier()


def test_vlm_verifier_loads(verifier) -> None:
    assert hasattr(verifier, "model")
    assert hasattr(verifier, "processor")


def test_verify_returns_result_type(verifier, sample_image_bgr: np.ndarray) -> None:
    from src.models.vlm_verifier import VerificationResult

    result = verifier.verify(sample_image_bgr, sample_image_bgr)
    assert isinstance(result, VerificationResult)
    assert isinstance(result.is_same, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.raw_output, str)


def test_confidence_in_valid_range(verifier, sample_image_bgr: np.ndarray) -> None:
    result = verifier.verify(sample_image_bgr, sample_image_bgr)
    assert 0.0 <= result.confidence <= 1.0, (
        f"confidence out of range: {result.confidence}"
    )


def test_same_image_is_same_person(verifier, sample_image_bgr: np.ndarray) -> None:
    result = verifier.verify(sample_image_bgr, sample_image_bgr)
    print(f"\n[smoke_vlm] same-image result: is_same={result.is_same}, confidence={result.confidence:.2f}")
    print(f"  reasoning: {result.reasoning}")
    assert result.is_same, (
        f"Expected is_same=True for identical images, got False. "
        f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
    )


def test_verify_saves_comparison(verifier, sample_image_bgr: np.ndarray) -> None:
    result = verifier.verify(sample_image_bgr, sample_image_bgr)

    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/smoke_vlm_verify.jpg")

    h, w = sample_image_bgr.shape[:2]
    compare = np.hstack([sample_image_bgr, sample_image_bgr])

    label = f"is_same={result.is_same} conf={result.confidence:.2f}"
    cv2.putText(compare, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(str(out_path), compare)
    print(f"\n[smoke_vlm] saved comparison to {out_path}")

    assert out_path.exists(), f"Output file not created: {out_path}"
