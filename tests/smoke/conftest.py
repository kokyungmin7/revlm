from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def sample_image_bgr() -> np.ndarray:
    """First jpg in WB_WoB-ReID_sample/. Skips if not found."""
    candidates = sorted(Path("WB_WoB-ReID_sample").glob("**/*.jpg"))
    # candidates = sorted(Path("WB_WoB-ReID_sample").glob("both_large/bounding_box_train/082_c4_f72.jpg"))

    if not candidates:
        pytest.skip("No sample image in WB_WoB-ReID_sample/")
    img = cv2.imread(str(candidates[0]))
    if img is None:
        pytest.skip("Failed to read sample image")
    return img
