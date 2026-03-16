from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def sample_image_bgr() -> np.ndarray:
    """First jpg in WB_WoB-ReID_sample/ or /home/kokyungmin/data/WB_WoB-ReID. Skips if not found."""
    search_roots = [
        Path("WB_WoB-ReID_sample"),
        Path("/home/kokyungmin/data/WB_WoB-ReID"),
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates = sorted(root.glob("**/*.jpg"))
            if candidates:
                break

    if not candidates:
        pytest.skip("No sample image found in WB_WoB-ReID_sample/ or /home/kokyungmin/data/WB_WoB-ReID")
    img = cv2.imread(str(candidates[0]))
    if img is None:
        pytest.skip("Failed to read sample image")
    return img
