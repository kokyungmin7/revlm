"""Run VLM verifier on dataset pairs and populate HITL queue.

Samples query images from bounding_box_train and verifies each against ALL
other positives (same person, different image) and N sampled negatives,
writing wrong predictions directly to labeled.jsonl for LoRA training.

Uses bounding_box_train for both query and gallery to guarantee person ID
overlap and to keep the test set completely unseen until evaluation.

Usage:
    uv run python scripts/run_hitl_inference.py [options]

Options:
    --data-root       Dataset root (default: /home/kokyungmin/data/WB_WoB-ReID)
    --hitl-dir        HITL data dir (default: data/hitl)
    --threshold       Queue predictions below this confidence (default: 0.7)
    --n-queries       Number of query images to process (default: 50)
    --n-negatives     Negative gallery images sampled per query (default: 5)
    --split           Dataset split subdir name, e.g. 'both_large' (default: auto-detect)
    --seed            Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2

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


def _build_index(pool: list[Path]) -> dict[str, list[Path]]:
    """Build person_id → image paths index from a flat image list.

    Args:
        pool: List of image paths following {id}_c{cam}_f{frame}.jpg naming.

    Returns:
        Dict mapping person ID string to list of image paths.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for p in pool:
        pid = _person_id(p)
        if pid is not None:
            index[pid].append(p)
    return dict(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate HITL queue via VLM inference.")
    parser.add_argument("--data-root", default="/home/kokyungmin/data/WB_WoB-ReID")
    parser.add_argument("--hitl-dir", default="data/hitl")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-negatives", type=int, default=5,
                        help="Negative gallery images sampled per query (default: 5)")
    parser.add_argument("--split", default="both_large",
                        choices=["both_large", "both_small", "with_bag", "without_bag"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: data root not found: {root}")
        return

    split_dir = _find_split_dir(root, args.split)
    print(f"Using split directory: {split_dir}")

    # HITL inference uses train gallery for BOTH query and gallery.
    # This guarantees person ID overlap and keeps test set unseen.
    train_gallery_dir = split_dir / "bounding_box_train"
    if not train_gallery_dir.exists():
        print(f"ERROR: bounding_box_train/ not found under {split_dir}")
        print("  HITL inference requires the train gallery to prevent test set contamination.")
        return
    pool = list(train_gallery_dir.glob("*.jpg"))
    if not pool:
        print(f"ERROR: bounding_box_train/ is empty under {split_dir}")
        return

    # Pre-build index: person_id → [image paths]
    id_to_images = _build_index(pool)
    all_ids = sorted(id_to_images.keys())

    # Query candidates: persons with >= 2 images (need at least one positive besides self)
    query_candidates = [
        img for pid in all_ids if len(id_to_images[pid]) >= 2
        for img in id_to_images[pid]
    ]
    if not query_candidates:
        print("ERROR: no person with >= 2 images in bounding_box_train/")
        return

    # Sample up to n_queries (seeded for reproducibility)
    random.seed(args.seed)
    sampled = random.sample(query_candidates, min(args.n_queries, len(query_candidates)))

    n_pos_total = sum(len(id_to_images.get(_person_id(q) or "", [])) - 1 for q in sampled)
    n_neg_total = len(sampled) * args.n_negatives
    print(f"Queries to process : {len(sampled)}")
    print(f"Query + Gallery    : bounding_box_train ({len(pool)} images, {len(all_ids)} identities)")
    print(f"Pairs planned      : ~{n_pos_total} positives + {n_neg_total} negatives")
    print(f"HITL threshold     : {args.threshold}")
    print(f"HITL directory     : {args.hitl_dir}\n")

    # Load verifier (heavy — CUDA required).
    # hitl_threshold=None disables confidence-based internal queuing —
    # we queue based on GT correctness instead.
    from src.models.vlm_verifier import load_vlm_verifier
    from src.models.hitl_collector import HITLCollector

    verifier = load_vlm_verifier(hitl_threshold=None)
    collector = HITLCollector(args.hitl_dir)

    auto_labeled = 0   # wrong predictions queued directly to labeled.jsonl
    processed = 0
    skipped = 0
    correct = 0
    total_pairs = 0

    SEP = "─" * 72

    for query_path in sampled:
        qid = _person_id(query_path)
        if qid is None or qid not in id_to_images:
            skipped += 1
            continue

        # Exclude self from positives
        positives = [p for p in id_to_images[qid] if p != query_path]
        neg_ids = [pid for pid in all_ids if pid != qid]
        if not positives or not neg_ids:
            skipped += 1
            continue

        # Sample negatives: pick n_negatives distinct IDs, one image each
        sampled_neg_ids = random.sample(neg_ids, min(args.n_negatives, len(neg_ids)))
        negatives = [random.choice(id_to_images[nid]) for nid in sampled_neg_ids]

        bgr_q = cv2.imread(str(query_path))
        if bgr_q is None:
            skipped += 1
            continue

        processed += 1
        h_q, w_q = bgr_q.shape[:2]
        n_pairs = len(positives) + len(negatives)

        print(SEP)
        print(
            f"[{processed}/{len(sampled)}] Query  person_id={qid}"
            f"  file={query_path.name}  size={w_q}x{h_q}"
            f"  ({len(positives)} pos + {len(negatives)} neg = {n_pairs} pairs)"
        )

        # ── Positive pairs (all, excluding self) ─────────────────────
        for pos_path in positives:
            bgr_p = cv2.imread(str(pos_path))
            if bgr_p is None:
                continue

            res = verifier.verify(bgr_q, bgr_p)
            total_pairs += 1
            gt_label = True
            is_wrong = res.is_same != gt_label
            if is_wrong:
                collector.log_labeled(bgr_q, bgr_p, res, gt_label)
                auto_labeled += 1
            else:
                correct += 1

            h_p, w_p = bgr_p.shape[:2]
            print(
                f"  [+] Positive  file={pos_path.name}  size={w_p}x{h_p}\n"
                f"      Prediction : {'SAME      ' if res.is_same else 'DIFFERENT '}"
                f" (GT: SAME)  {'✓' if not is_wrong else '✗'}"
                f"  conf={res.confidence:.3f}"
                + (f"  ← AUTO-LABELED" if is_wrong else "")
            )

        # ── Negative pairs (sampled) ────────────────────────────────────
        for neg_path in negatives:
            nid = _person_id(neg_path)
            bgr_n = cv2.imread(str(neg_path))
            if bgr_n is None:
                continue

            res = verifier.verify(bgr_q, bgr_n)
            total_pairs += 1
            gt_label = False
            is_wrong = res.is_same != gt_label
            if is_wrong:
                collector.log_labeled(bgr_q, bgr_n, res, gt_label)
                auto_labeled += 1
            else:
                correct += 1

            h_n, w_n = bgr_n.shape[:2]
            print(
                f"  [-] Negative  person_id={nid}  file={neg_path.name}  size={w_n}x{h_n}\n"
                f"      Prediction : {'SAME      ' if res.is_same else 'DIFFERENT '}"
                f" (GT: DIFFERENT)  {'✓' if not is_wrong else '✗'}"
                f"  conf={res.confidence:.3f}"
                + (f"  ← AUTO-LABELED" if is_wrong else "")
            )

    accuracy = correct / total_pairs if total_pairs > 0 else 0.0

    print(SEP)
    print("\n=== Run Summary ===")
    print(f"  Queries processed  : {processed}  (skipped: {skipped})")
    print(f"  Pairs evaluated    : {total_pairs}")
    print(f"  VLM accuracy       : {correct}/{total_pairs} = {accuracy:.1%}")
    print(f"  Wrong → auto-labeled : {auto_labeled}  (written to labeled.jsonl)")
    print(f"  Total labeled      : {collector.labeled_size}")
    if auto_labeled > 0:
        print(f"\nNext step: uv run python scripts/lora_train.py --min-samples {auto_labeled}")


if __name__ == "__main__":
    main()
