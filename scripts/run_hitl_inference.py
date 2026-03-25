"""Run VLM verifier on dataset pairs and populate HITL queue.

Iterates over query images in WB_WoB-ReID, verifies each against a
positive and a negative, and queues low-confidence results for HITL review.

Usage:
    uv run python scripts/run_hitl_inference.py [options]

Options:
    --data-root     Dataset root (default: /home/kokyungmin/data/WB_WoB-ReID)
    --hitl-dir      HITL data dir (default: data/hitl)
    --threshold     Queue predictions below this confidence (default: 0.7)
    --n-queries     Number of query images to process (default: 50)
    --split         Dataset split subdir name, e.g. 'both_large' (default: auto-detect)
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

_ID_RE = re.compile(r"^(\d+)_c\d+_f\d+\.jpg$", re.IGNORECASE)


def _person_id(path: Path) -> str | None:
    m = _ID_RE.match(path.name)
    return m.group(1) if m else None


def _find_split_dir(root: Path, split: str | None) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate HITL queue via VLM inference.")
    parser.add_argument("--data-root", default="/home/kokyungmin/data/WB_WoB-ReID")
    parser.add_argument("--hitl-dir", default="data/hitl")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: data root not found: {root}")
        return

    split_dir = _find_split_dir(root, args.split)
    print(f"Using split directory: {split_dir}")

    query_images = sorted((split_dir / "query").glob("*.jpg"))
    if not query_images:
        print(f"ERROR: no images in {split_dir}/query/")
        return

    pool = (
        list((split_dir / "bounding_box_train").glob("*.jpg"))
        + list((split_dir / "bounding_box_test").glob("*.jpg"))
    )

    # Sample up to n_queries
    sampled = random.sample(query_images, min(args.n_queries, len(query_images)))

    print(f"Queries to process : {len(sampled)}")
    print(f"HITL threshold     : {args.threshold}")
    print(f"HITL directory     : {args.hitl_dir}\n")

    # Load verifier (heavy — CUDA required)
    from src.models.vlm_verifier import load_vlm_verifier

    verifier = load_vlm_verifier(
        hitl_threshold=args.threshold,
        hitl_data_dir=args.hitl_dir,
    )

    queued = 0
    processed = 0

    for query_path in sampled:
        qid = _person_id(query_path)
        if qid is None:
            continue

        positives = [p for p in pool if _person_id(p) == qid]
        negatives = [p for p in pool if _person_id(p) != qid and _person_id(p) is not None]
        if not positives or not negatives:
            continue

        pos_path = random.choice(positives)
        neg_path = random.choice(negatives)

        bgr_q = cv2.imread(str(query_path))
        bgr_p = cv2.imread(str(pos_path))
        bgr_n = cv2.imread(str(neg_path))

        if bgr_q is None or bgr_p is None or bgr_n is None:
            continue

        # Positive pair
        res_pos = verifier.verify(bgr_q, bgr_p)
        if res_pos.confidence < args.threshold:
            queued += 1
        print(
            f"[{processed+1}/{len(sampled)}] +pair id={qid} "
            f"same={res_pos.is_same} conf={res_pos.confidence:.3f}"
            + (" → QUEUED" if res_pos.confidence < args.threshold else "")
        )

        # Negative pair
        res_neg = verifier.verify(bgr_q, bgr_n)
        if res_neg.confidence < args.threshold:
            queued += 1
        print(
            f"[{processed+1}/{len(sampled)}] -pair id={qid}  "
            f"same={res_neg.is_same} conf={res_neg.confidence:.3f}"
            + (" → QUEUED" if res_neg.confidence < args.threshold else "")
        )

        processed += 1

    from src.models.hitl_collector import HITLCollector
    collector = HITLCollector(args.hitl_dir)
    print(f"\nDone. Processed {processed} queries ({processed * 2} pairs).")
    print(f"Queued for review : {collector.queue_size}")
    print(f"\nNext step: uv run python scripts/hitl_review.py --data-dir {args.hitl_dir}")


if __name__ == "__main__":
    main()
