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
    skipped = 0
    correct = 0   # VLM prediction matches ground truth
    total_pairs = 0

    SEP = "─" * 72

    for query_path in sampled:
        qid = _person_id(query_path)
        if qid is None:
            skipped += 1
            continue

        positives = [p for p in pool if _person_id(p) == qid]
        negatives = [p for p in pool if _person_id(p) != qid and _person_id(p) is not None]
        if not positives or not negatives:
            skipped += 1
            continue

        pos_path = random.choice(positives)
        neg_path = random.choice(negatives)

        bgr_q = cv2.imread(str(query_path))
        bgr_p = cv2.imread(str(pos_path))
        bgr_n = cv2.imread(str(neg_path))

        if bgr_q is None or bgr_p is None or bgr_n is None:
            skipped += 1
            continue

        processed += 1
        h_q, w_q = bgr_q.shape[:2]
        h_p, w_p = bgr_p.shape[:2]
        h_n, w_n = bgr_n.shape[:2]

        print(SEP)
        print(
            f"[{processed}/{len(sampled)}] Query  person_id={qid}"
            f"  file={query_path.name}  size={w_q}x{h_q}"
        )

        # ── Positive pair ──────────────────────────────────────────────
        res_pos = verifier.verify(bgr_q, bgr_p)
        total_pairs += 1
        is_queued_pos = res_pos.confidence < args.threshold
        if is_queued_pos:
            queued += 1
        gt_correct_pos = res_pos.is_same  # ground truth: same
        if gt_correct_pos:
            correct += 1

        print(
            f"  [+] Positive  person_id={qid}"
            f"  file={pos_path.name}  size={w_p}x{h_p}"
        )
        print(
            f"      Prediction : {'SAME      ' if res_pos.is_same else 'DIFFERENT '}"
            f" (GT: SAME)  {'✓ correct' if gt_correct_pos else '✗ wrong'}"
        )
        print(f"      Confidence : {res_pos.confidence:.3f}"
              + (f"  ← QUEUED (< {args.threshold})" if is_queued_pos else ""))
        print(f"      Reasoning  : {res_pos.reasoning}")

        # ── Negative pair ──────────────────────────────────────────────
        nid = _person_id(neg_path)
        res_neg = verifier.verify(bgr_q, bgr_n)
        total_pairs += 1
        is_queued_neg = res_neg.confidence < args.threshold
        if is_queued_neg:
            queued += 1
        gt_correct_neg = not res_neg.is_same  # ground truth: different
        if gt_correct_neg:
            correct += 1

        print(
            f"  [-] Negative  person_id={nid}"
            f"  file={neg_path.name}  size={w_n}x{h_n}"
        )
        print(
            f"      Prediction : {'SAME      ' if res_neg.is_same else 'DIFFERENT '}"
            f" (GT: DIFFERENT)  {'✓ correct' if gt_correct_neg else '✗ wrong'}"
        )
        print(f"      Confidence : {res_neg.confidence:.3f}"
              + (f"  ← QUEUED (< {args.threshold})" if is_queued_neg else ""))
        print(f"      Reasoning  : {res_neg.reasoning}")

    from src.models.hitl_collector import HITLCollector
    collector = HITLCollector(args.hitl_dir)

    accuracy = correct / total_pairs if total_pairs > 0 else 0.0

    print(SEP)
    print("\n=== Run Summary ===")
    print(f"  Queries processed : {processed}  (skipped: {skipped})")
    print(f"  Pairs evaluated   : {total_pairs}  (positive: {processed}, negative: {processed})")
    print(f"  VLM accuracy      : {correct}/{total_pairs} = {accuracy:.1%}")
    print(f"  Queued for review : {queued}  (confidence < {args.threshold})")
    print(f"  Total in queue    : {collector.queue_size}")
    print(f"  Total labeled     : {collector.labeled_size}")
    if collector.queue_size > 0:
        print(f"\nNext step: uv run python scripts/hitl_review.py --data-dir {args.hitl_dir}")


if __name__ == "__main__":
    main()
