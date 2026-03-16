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

    # split 서브디렉토리 감지: root/query/ 없으면 root/{split}/query/ 탐색
    split_dir: Path
    if (root / "query").exists():
        split_dir = root
    else:
        splits = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "query").exists()]
        if not splits:
            pytest.skip("No split directory with query/ found")
        split_dir = random.choice(splits)

    query_images = sorted((split_dir / "query").glob("*.jpg"))
    if not query_images:
        pytest.skip(f"No images in {split_dir}/query/")

    query_path = random.choice(query_images)
    qid = _person_id(query_path)
    if qid is None:
        pytest.skip(f"Cannot parse person ID from {query_path.name}")

    pool = list((split_dir / "bounding_box_train").glob("*.jpg")) + \
           list((split_dir / "bounding_box_test").glob("*.jpg"))

    positives = [p for p in pool if _person_id(p) == qid]
    negatives = [p for p in pool if _person_id(p) != qid and _person_id(p) is not None]

    if not positives:
        pytest.skip(f"No positive images for ID {qid} in {split_dir.name}")
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
