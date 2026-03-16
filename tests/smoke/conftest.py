import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

_ID_RE = re.compile(r"^(\d+)_c\d+_f\d+\.jpg$", re.IGNORECASE)


def _person_id(path: Path) -> str | None:
    m = _ID_RE.match(path.name)
    return m.group(1) if m else None


def _find_data_root() -> Path | None:
    for root in [Path("WB_WoB-ReID_sample"), Path("/home/kokyungmin/data/WB_WoB-ReID")]:
        if root.exists():
            return root
    return None


@dataclass
class ReidTriplet:
    query: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    query_id: str
    positive_id: str
    negative_id: str


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


@pytest.fixture(scope="module")
def reid_triplet() -> ReidTriplet:
    """Random query/positive/negative triplet from WB_WoB-ReID dataset. Skips if not found."""
    root = _find_data_root()
    if root is None:
        pytest.skip("No dataset found")

    query_images = sorted((root / "query").glob("*.jpg"))
    if not query_images:
        pytest.skip("No images in query/")

    query_path = random.choice(query_images)
    qid = _person_id(query_path)
    if qid is None:
        pytest.skip(f"Cannot parse person ID from {query_path.name}")

    pool = list((root / "bounding_box_train").glob("*.jpg")) + \
           list((root / "bounding_box_test").glob("*.jpg"))

    positives = [p for p in pool if _person_id(p) == qid]
    negatives = [p for p in pool if _person_id(p) != qid and _person_id(p) is not None]

    if not positives:
        pytest.skip(f"No positive images for ID {qid}")
    if not negatives:
        pytest.skip("No negative images found")

    def read(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            pytest.skip(f"Failed to read {p}")
        return img

    pos_path = random.choice(positives)
    neg_path = random.choice(negatives)

    return ReidTriplet(
        query=read(query_path),
        positive=read(pos_path),
        negative=read(neg_path),
        query_id=qid,
        positive_id=_person_id(pos_path),
        negative_id=_person_id(neg_path),
    )
