"""Fixed evaluation pair set for reproducible snowball loop benchmarking.

Builds a balanced set of (positive, negative) pairs from WB_WoB-ReID
and persists them to JSONL so all three pipeline stages are evaluated
on identical inputs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_ID_RE = re.compile(r"^(\d+)_c\d+_f\d+\.jpg$", re.IGNORECASE)


def _person_id(path: Path) -> str | None:
    m = _ID_RE.match(path.name)
    return m.group(1) if m else None


def _find_split_dir(root: Path, split: str | None) -> Path:
    """Locate the split subdirectory that contains a query/ folder."""
    if (root / "query").exists():
        return root
    if split:
        candidate = root / split
        if (candidate / "query").exists():
            return candidate
        raise FileNotFoundError(f"Split '{split}' not found under {root}")
    splits = sorted(d for d in root.iterdir() if d.is_dir() and (d / "query").exists())
    if not splits:
        raise FileNotFoundError(f"No split with query/ found under {root}")
    return splits[0]


@dataclass
class EvalPair:
    """A single evaluation pair with ground truth label.

    Attributes:
        img_path_a: Absolute path to the query image.
        img_path_b: Absolute path to the gallery image.
        label: True = same person, False = different person.
        person_id_a: Person ID of img_a.
        person_id_b: Person ID of img_b.
    """

    img_path_a: str
    img_path_b: str
    label: bool
    person_id_a: str
    person_id_b: str


def build_eval_pairs(
    data_root: str,
    n_queries: int = 100,
    seed: int = 42,
    split: str | None = None,
) -> list[EvalPair]:
    """Build a balanced list of positive and negative evaluation pairs.

    For each of n_queries query images, one positive (same person) and one
    negative (different person) pair is created, yielding 2 * n_queries pairs
    total with a 50/50 label balance.

    Args:
        data_root: Root directory of WB_WoB-ReID dataset.
        n_queries: Number of query images to sample.
        seed: Random seed for reproducibility.
        split: Optional split subdirectory name (e.g., 'both_large').

    Returns:
        List of EvalPair with alternating positive/negative pairs.

    Raises:
        FileNotFoundError: If the data root or split directory is not found.
        ValueError: If not enough images are available.
    """
    rng = random.Random(seed)
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    split_dir = _find_split_dir(root, split)
    query_images = sorted((split_dir / "query").glob("*.jpg"))
    if not query_images:
        raise ValueError(f"No images in {split_dir}/query/")

    # Evaluation uses test gallery only — never train images.
    pool = list((split_dir / "bounding_box_test").glob("*.jpg"))

    sampled_queries = rng.sample(query_images, min(n_queries, len(query_images)))

    pairs: list[EvalPair] = []
    for query_path in sampled_queries:
        qid = _person_id(query_path)
        if qid is None:
            continue

        positives = [p for p in pool if _person_id(p) == qid]
        negatives = [p for p in pool if _person_id(p) != qid and _person_id(p) is not None]
        if not positives or not negatives:
            continue

        pos_path = rng.choice(positives)
        neg_path = rng.choice(negatives)

        pairs.append(
            EvalPair(
                img_path_a=str(query_path),
                img_path_b=str(pos_path),
                label=True,
                person_id_a=qid,
                person_id_b=qid,
            )
        )
        pairs.append(
            EvalPair(
                img_path_a=str(query_path),
                img_path_b=str(neg_path),
                label=False,
                person_id_a=qid,
                person_id_b=_person_id(neg_path) or "unknown",
            )
        )

    return pairs


def save_eval_pairs(pairs: list[EvalPair], path: str) -> None:
    """Persist evaluation pairs to a JSONL file.

    Args:
        pairs: List of EvalPair instances.
        path: Destination file path (created if not exists).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
    print(f"Saved {len(pairs)} eval pairs → {out}")


def load_eval_pairs(path: str) -> list[EvalPair]:
    """Load evaluation pairs from a JSONL file.

    Args:
        path: Path to JSONL file produced by save_eval_pairs.

    Returns:
        List of EvalPair instances.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval pairs file not found: {p}")
    pairs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(EvalPair(**json.loads(line)))
    return pairs
