"""Extract MEBOW orientation angles and ReID embeddings from WB/WoB-ReID dataset.

Usage:
    # 전체 4개 세트 추출 (GCP)
    uv run python experiments/extract_embeddings.py \\
        --dataset_root /home/kokyungmin/data/WB_WoB-ReID \\
        --sets with_bag without_bag both_large both_small \\
        --output experiments/outputs/embeddings.json \\
        --mebow_root ./third_party/MEBOW \\
        --device cuda

    # 로컬 샘플 테스트
    uv run python experiments/extract_embeddings.py \\
        --dataset_root ./WB_WoB-ReID_sample \\
        --sets both_large \\
        --output /tmp/test_embeddings.json

Notes:
    - Images are already cropped (no detection step).
    - person_id <= 164 → bag_set = "with_bag_person"
    - person_id >= 165 → bag_set = "without_bag_person"
    - Filename format: {person_id}_c{camera_id}_f{frame_no}.jpg
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is importable (same convention as tests/smoke/)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.models.reid import extract_embedding, load_reid_model
from src.preprocessing.body_orientation import load_mebow_model, predict_orientation

_SPLITS = ("bounding_box_train", "bounding_box_test", "query")
_FILENAME_RE = re.compile(r"^(\d+)_c(\d+)_f(\d+)\.jpg$", re.IGNORECASE)


def _parse_filename(filename: str) -> tuple[int, int, int] | None:
    """Parse person_id, camera_id, frame_no from filename.

    Handles both zero-padded (0001) and non-padded (083) person IDs.

    Returns:
        (person_id, camera_id, frame_no) as ints, or None if pattern does not match.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _bag_set(person_id: int) -> str:
    """Determine which original set the person belongs to.

    with_bag set covers persons 1-164; without_bag covers 165-500.
    """
    return "with_bag_person" if person_id <= 164 else "without_bag_person"


def _collect_images(
    dataset_root: Path,
    sets: list[str],
) -> list[dict]:
    """Collect all image paths and parsed metadata.

    Args:
        dataset_root: Root of WB_WoB-ReID dataset.
        sets: List of set names (e.g. ["with_bag", "both_large"]).

    Returns:
        List of dicts with keys: path, filename, set, split, person_id,
        camera_id, frame_no, bag_set.
    """
    records: list[dict] = []

    for set_name in sets:
        for split in _SPLITS:
            split_dir = dataset_root / set_name / split
            if not split_dir.exists():
                continue
            for img_path in sorted(split_dir.glob("*.jpg")):
                parsed = _parse_filename(img_path.name)
                if parsed is None:
                    continue
                person_id, camera_id, frame_no = parsed
                rel_path = str(img_path.relative_to(dataset_root))
                records.append(
                    {
                        "filename": img_path.name,
                        "path": rel_path,
                        "set": set_name,
                        "split": split,
                        "person_id": person_id,
                        "camera_id": camera_id,
                        "frame_no": frame_no,
                        "bag_set": _bag_set(person_id),
                    }
                )

    return records


def _load_bgr(image_path: Path) -> np.ndarray | None:
    """Load image as BGR ndarray, return None on failure."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MEBOW angles and ReID embeddings from WB/WoB-ReID dataset."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=Path,
        help="Root directory of the WB_WoB-ReID dataset.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["with_bag", "without_bag", "both_large", "both_small"],
        choices=["with_bag", "without_bag", "both_large", "both_small"],
        help="Dataset sets to process.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--mebow_root",
        default="./third_party/MEBOW",
        type=Path,
        help="Path to cloned MEBOW repository.",
    )
    parser.add_argument(
        "--mebow_cfg",
        default=None,
        type=Path,
        help="MEBOW config YAML. Defaults to experiments/coco/segm-4_lr1e-3.yaml inside mebow_root.",
    )
    parser.add_argument(
        "--mebow_weights",
        default=None,
        type=Path,
        help="MEBOW model weights (.pth). Defaults to cfg.TEST.MODEL_FILE.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda / cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--reid_model",
        default="arcface-dinov2",
        choices=["arcface-dinov2", "siglip2"],
        help="ReID backend to use (default: arcface-dinov2).",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    output_path: Path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── collect image records ─────────────────────────────────────────────────
    print(f"[1/3] Collecting images from {dataset_root} ...")
    records = _collect_images(dataset_root, args.sets)
    print(f"      {len(records)} images found across sets: {args.sets}")

    if not records:
        print("No images found. Check --dataset_root and --sets.", file=sys.stderr)
        sys.exit(1)

    # ── load models ───────────────────────────────────────────────────────────
    print("[2/3] Loading models ...")
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reid_model = load_reid_model(name=args.reid_model, device=device)
    print(f"      ReID model '{args.reid_model}' loaded (embed_dim={reid_model.embed_dim})")

    mebow_cfg = str(args.mebow_cfg) if args.mebow_cfg else None
    mebow_kwargs: dict = dict(
        mebow_root=str(args.mebow_root),
        cfg_path=mebow_cfg,
        device=device,
    )
    if args.mebow_weights:
        mebow_kwargs["model_path"] = str(args.mebow_weights)
    mebow_model, _ = load_mebow_model(**mebow_kwargs)
    print("      MEBOW model loaded")

    # ── extract features ──────────────────────────────────────────────────────
    print("[3/3] Extracting embeddings ...")
    failed = 0
    results: list[dict] = []

    for rec in tqdm(records, unit="img"):
        img_path = dataset_root / rec["path"]
        bgr = _load_bgr(img_path)
        if bgr is None:
            failed += 1
            continue

        angle = predict_orientation(mebow_model, bgr, device)
        embedding = extract_embedding(reid_model, bgr, device)

        result = dict(rec)
        result["orientation_angle"] = angle
        result["embedding"] = embedding.tolist()
        results.append(result)

    print(f"      Done. {len(results)} succeeded, {failed} failed.")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "dataset_root": str(dataset_root),
            "sets": args.sets,
            "models": {
                "reid": args.reid_model,
                "orientation": "MEBOW-HRNet-W32",
            },
            "embedding_dim": reid_model.embed_dim,
            "total_images": len(results),
            "failed_images": failed,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        },
        "images": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
