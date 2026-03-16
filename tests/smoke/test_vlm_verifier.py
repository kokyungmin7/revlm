"""Smoke tests for VLM-based ReID verification module.

Requires CUDA environment (Qwen3-VL-8B-Instruct, ~16GB bfloat16).
Run on GCP L4 or equivalent CUDA GPU.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tests.smoke.conftest import ReidTriplet

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


def test_verify_returns_result_type(verifier, reid_triplet: ReidTriplet) -> None:
    from src.models.vlm_verifier import VerificationResult

    result = verifier.verify(reid_triplet.query, reid_triplet.positive)
    assert isinstance(result, VerificationResult)
    assert isinstance(result.is_same, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.raw_output, str)


def test_confidence_in_valid_range(verifier, reid_triplet: ReidTriplet) -> None:
    result = verifier.verify(reid_triplet.query, reid_triplet.positive)
    assert 0.0 <= result.confidence <= 1.0, (
        f"confidence out of range: {result.confidence}"
    )


def test_same_person_is_same(verifier, reid_triplet: ReidTriplet) -> None:
    result = verifier.verify(reid_triplet.query, reid_triplet.positive)
    print(f"\n[smoke_vlm] same-person result: is_same={result.is_same}, confidence={result.confidence:.2f}")
    print(f"  IDs: query={reid_triplet.query_id}, positive={reid_triplet.positive_id}")
    print(f"  reasoning: {result.reasoning}")
    assert result.is_same, (
        f"Expected is_same=True for same person (query={reid_triplet.query_id}, "
        f"positive={reid_triplet.positive_id}), got False. "
        f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
    )


def test_different_person_is_different(verifier, reid_triplet: ReidTriplet) -> None:
    result = verifier.verify(reid_triplet.query, reid_triplet.negative)
    print(f"\n[smoke_vlm] diff-person: is_same={result.is_same}, conf={result.confidence:.2f}")
    print(f"  IDs: query={reid_triplet.query_id}, negative={reid_triplet.negative_id}")
    assert not result.is_same, (
        f"Expected is_same=False for different people (query={reid_triplet.query_id}, "
        f"negative={reid_triplet.negative_id}), got True. "
        f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
    )


def test_verify_saves_comparison(verifier, reid_triplet: ReidTriplet) -> None:
    result_pos = verifier.verify(reid_triplet.query, reid_triplet.positive)
    result_neg = verifier.verify(reid_triplet.query, reid_triplet.negative)

    Path("experiments").mkdir(exist_ok=True)
    out_path = Path("experiments/smoke_vlm_verify.jpg")

    target_h = 256

    def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    q = resize_to_height(reid_triplet.query, target_h)
    p = resize_to_height(reid_triplet.positive, target_h)
    n = resize_to_height(reid_triplet.negative, target_h)

    def add_label(img: np.ndarray, label: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out

    q = add_label(q, f"query id={reid_triplet.query_id}")
    p = add_label(p, f"pos id={reid_triplet.positive_id} same={result_pos.is_same}")
    n = add_label(n, f"neg id={reid_triplet.negative_id} same={result_neg.is_same}")

    compare = np.hstack([q, p, n])
    cv2.imwrite(str(out_path), compare)
    print(f"\n[smoke_vlm] saved 3-panel comparison to {out_path}")

    assert out_path.exists(), f"Output file not created: {out_path}"
