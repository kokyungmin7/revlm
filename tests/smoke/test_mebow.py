from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

MEBOW_ROOT = Path("third_party/MEBOW")
MODEL_PATH = Path("models/body_orientation/model_hboe.pth")

if not MEBOW_ROOT.exists():
    pytest.skip("MEBOW root not found. Run setup_mebow.py", allow_module_level=True)
if not MODEL_PATH.exists():
    pytest.skip("MEBOW weights not found", allow_module_level=True)


def test_mebow_returns_valid_angle(sample_image_bgr: np.ndarray) -> None:
    from src.preprocessing.body_orientation import load_mebow_model, predict_orientation

    model, device = load_mebow_model(
        mebow_root=str(MEBOW_ROOT),
        model_path=str(MODEL_PATH),
    )
    angle = predict_orientation(model, sample_image_bgr, device)

    valid_angles = set(range(0, 360, 5))
    assert angle in valid_angles, (
        f"Angle {angle} not in valid set (multiples of 5, 0~355)"
    )

    from pathlib import Path

    import cv2

    print(f"\n[smoke_mebow] predicted angle: {angle}°")

    Path("experiments").mkdir(exist_ok=True)
    vis = sample_image_bgr.copy()
    cv2.putText(
        vis,
        f"{angle} deg",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imwrite("experiments/smoke_mebow_angle.jpg", vis)
    print("  → saved: experiments/smoke_mebow_angle.jpg")
