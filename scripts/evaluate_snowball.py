"""Evaluate the ReID accuracy snowball loop across three pipeline stages.

Outputs a comparison table showing progressive accuracy improvement:
  Stage 1 — ReID backbone (cosine similarity, auto-threshold)
  Stage 2 — VLM verifier (base Qwen3-VL-8B-Instruct)
  Stage 3 — VLM verifier + HITL LoRA fine-tuned adapter

Usage:
    uv run python scripts/evaluate_snowball.py [options]

Options:
    --data-root       Dataset root (default: /home/kokyungmin/data/WB_WoB-ReID)
    --eval-pairs      JSONL file for fixed eval set (auto-built if missing)
    --n-queries       Queries when building eval set (default: 100)
    --split           Dataset split subdir (default: auto-detect)
    --reid-model      arcface-dinov2 | siglip2 (default: arcface-dinov2)
    --hitl-threshold  Confidence below which VLM defers to HITL (default: 0.7)
    --lora-adapter    Path to LoRA adapter dir for stage 3 (default: models/vlm_verifier_lora/latest)
    --skip-stage      1|2|3  comma-separated stages to skip (e.g. --skip-stage 3)
    --output          JSON file to save results (default: experiments/results/snowball.json)
    --seed            Random seed for eval set construction (default: 42)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ── Formatting helpers ────────────────────────────────────────────────────────

SEP_WIDE = "=" * 72
SEP_THIN = "─" * 72


def _fmt_pct(v: float) -> str:
    return f"{v * 100:6.2f}%"


def _print_stage_header(n: int, title: str) -> None:
    print(f"\n{'─'*4} Stage {n}: {title} {'─'*(50 - len(title))}")


def _print_pair_result(
    idx: int,
    total: int,
    img_a: str,
    img_b: str,
    id_a: str,
    id_b: str,
    label: bool,
    prediction: bool,
    confidence: float | None = None,
    reasoning: str | None = None,
    queued: bool = False,
) -> None:
    correct = prediction == label
    verdict = "SAME     " if prediction else "DIFFERENT"
    gt = "SAME     " if label else "DIFFERENT"
    mark = "✓" if correct else "✗"

    conf_str = f"  conf={confidence:.3f}" if confidence is not None else ""
    queued_str = "  ← QUEUED" if queued else ""

    print(
        f"  [{idx:>3}/{total}] {mark} "
        f"A={Path(img_a).name} (id={id_a})  "
        f"B={Path(img_b).name} (id={id_b})"
    )
    print(
        f"           pred={verdict}  gt={gt}{conf_str}{queued_str}"
    )
    if reasoning:
        print(f"           reason: {reasoning}")


def _print_metrics(m, threshold: float | None = None) -> None:
    from src.evaluation.metrics import EvalMetrics
    th_str = f"  threshold={threshold:.4f}" if threshold is not None else ""
    print(f"  Accuracy  : {_fmt_pct(m.accuracy)}  ({m.n_correct}/{m.n_total}){th_str}")
    print(f"  Precision : {_fmt_pct(m.precision)}   Recall: {_fmt_pct(m.recall)}   F1: {_fmt_pct(m.f1)}")
    print(f"  TP={m.tp}  TN={m.tn}  FP={m.fp}  FN={m.fn}")


def _print_summary_table(stages: list[dict]) -> None:
    print(f"\n{SEP_WIDE}")
    print("  ReID Accuracy Snowball Loop — Summary")
    print(SEP_WIDE)
    header = f"  {'Stage':<38} {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}"
    print(header)
    print(SEP_THIN)
    for s in stages:
        m = s["metrics"]
        print(
            f"  {s['label']:<38} "
            f"{_fmt_pct(m.accuracy)}  "
            f"{_fmt_pct(m.precision)}  "
            f"{_fmt_pct(m.recall)}  "
            f"{_fmt_pct(m.f1)}"
        )
    print(SEP_THIN)
    accs = [s["metrics"].accuracy for s in stages]
    if len(accs) >= 2:
        total_gain = accs[-1] - accs[0]
        print(f"  Overall improvement (stage 1 → {len(stages)}): {total_gain*100:+.2f}pp")
        for i in range(1, len(stages)):
            gain = accs[i] - accs[i - 1]
            print(f"  Stage {i} → {i+1}: {gain*100:+.2f}pp")
    print(SEP_WIDE)


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage1_reid(pairs, reid_model_name: str, threshold: float = 0.5) -> tuple[dict, list[bool]]:
    """Stage 1: ReID cosine similarity with a fixed threshold.

    Uses a pre-defined threshold instead of grid-searching on the test set,
    which would leak test information and inflate Stage 1 accuracy.
    """
    from src.models.reid import load_reid_model, compute_similarity
    from src.evaluation.metrics import compute_metrics

    _print_stage_header(1, f"ReID baseline ({reid_model_name})")
    print(f"  Loading model: {reid_model_name} ...")
    print(f"  Threshold    : {threshold} (fixed)")
    model = load_reid_model(reid_model_name)

    similarities: list[float] = []
    labels: list[bool] = []

    t0 = time.time()
    for i, pair in enumerate(pairs, 1):
        bgr_a = cv2.imread(pair.img_path_a)
        bgr_b = cv2.imread(pair.img_path_b)
        if bgr_a is None or bgr_b is None:
            print(f"  [WARN] Could not read image for pair {i}, skipping.")
            continue

        emb_a = model.extract_embedding(bgr_a)
        emb_b = model.extract_embedding(bgr_b)
        sim = compute_similarity(emb_a, emb_b)
        similarities.append(sim)
        labels.append(pair.label)

    predictions = [s >= threshold for s in similarities]
    metrics = compute_metrics(predictions, labels)

    elapsed = time.time() - t0
    print(f"  Processed {len(pairs)} pairs in {elapsed:.1f}s")
    _print_metrics(metrics, threshold=threshold)

    for i, (pair, pred, sim) in enumerate(zip(pairs, predictions, similarities), 1):
        _print_pair_result(
            i, len(pairs),
            pair.img_path_a, pair.img_path_b,
            pair.person_id_a, pair.person_id_b,
            pair.label, pred,
            confidence=sim,
        )

    return {
        "stage": 1,
        "label": f"1. ReID only ({reid_model_name})",
        "model": reid_model_name,
        "threshold": threshold,
        "metrics": metrics,
        "elapsed_s": elapsed,
    }, predictions


def _run_vlm_stage(
    stage_n: int,
    stage_title: str,
    pairs,
    hitl_threshold: float,
    batch_size: int,
    lora_adapter_path: str | None = None,
) -> tuple[dict, list[bool]]:
    """Shared implementation for Stage 2 and Stage 3 VLM evaluation.

    Args:
        stage_n: Stage number (2 or 3).
        stage_title: Human-readable stage title.
        pairs: List of EvalPair to evaluate.
        hitl_threshold: Confidence threshold reported in HITL stats (no actual queuing).
        batch_size: Number of pairs per VLM forward pass.
        lora_adapter_path: LoRA adapter directory, or None for base model.
    """
    from src.models.vlm_verifier import load_vlm_verifier
    from src.evaluation.metrics import compute_metrics

    _print_stage_header(stage_n, stage_title)
    model_desc = "Qwen3-VL-8B-Instruct"
    if lora_adapter_path:
        print(f"  Loading {model_desc} + LoRA from: {lora_adapter_path}")
    else:
        print(f"  Loading {model_desc} ...")
    print(f"  Batch size : {batch_size}")

    # hitl_threshold=None: evaluation must NOT write to HITL queue (test data contamination)
    verifier = load_vlm_verifier(hitl_threshold=None, lora_adapter_path=lora_adapter_path)

    # Read all images upfront to filter missing files
    valid_pairs = []
    valid_bgr = []
    for pair in pairs:
        bgr_a = cv2.imread(pair.img_path_a)
        bgr_b = cv2.imread(pair.img_path_b)
        if bgr_a is None or bgr_b is None:
            print(f"  [WARN] Could not read images for pair ({pair.img_path_a}), skipping.")
            continue
        valid_pairs.append(pair)
        valid_bgr.append((bgr_a, bgr_b))

    predictions: list[bool] = []
    results_all: list = []
    n_queued = 0
    pair_idx = 0

    t0 = time.time()
    for batch_start in range(0, len(valid_bgr), batch_size):
        batch_bgr = valid_bgr[batch_start: batch_start + batch_size]
        batch_end = batch_start + len(batch_bgr)
        print(f"  Inferring [{batch_end}/{len(valid_bgr)}] ...", flush=True)

        batch_results = verifier.verify_batch(batch_bgr)
        results_all.extend(batch_results)

        for r in batch_results:
            predictions.append(r.is_same)
            pair = valid_pairs[pair_idx]
            queued = r.confidence < hitl_threshold
            if queued:
                n_queued += 1
            _print_pair_result(
                pair_idx + 1, len(valid_pairs),
                pair.img_path_a, pair.img_path_b,
                pair.person_id_a, pair.person_id_b,
                pair.label, r.is_same,
                confidence=r.confidence,
                reasoning=r.reasoning,
                queued=queued,
            )
            pair_idx += 1

    labels = [p.label for p in valid_pairs]

    metrics = compute_metrics(predictions, labels)
    elapsed = time.time() - t0
    print(f"\n  Processed {len(valid_pairs)} pairs in {elapsed:.1f}s  (batch_size={batch_size})")
    if stage_n == 2:
        print(f"  Would queue for HITL : {n_queued} pairs (confidence < {hitl_threshold})")
    _print_metrics(metrics)

    row: dict = {
        "stage": stage_n,
        "label": f"{stage_n}. {stage_title}",
        "model": model_desc,
        "batch_size": batch_size,
        "metrics": metrics,
        "elapsed_s": elapsed,
    }
    if stage_n == 2:
        row["hitl_threshold"] = hitl_threshold
        row["n_would_queue"] = n_queued
    if lora_adapter_path:
        row["lora_adapter"] = lora_adapter_path
    return row, predictions


def run_stage2_vlm(pairs, hitl_threshold: float, batch_size: int = 1) -> tuple[dict, list[bool]]:
    """Stage 2: VLM verifier (base model, no LoRA)."""
    return _run_vlm_stage(
        stage_n=2,
        stage_title="VLM verifier (base)",
        pairs=pairs,
        hitl_threshold=hitl_threshold,
        batch_size=batch_size,
    )


def run_stage3_vlm_lora(
    pairs, lora_adapter_path: str, hitl_threshold: float, batch_size: int = 1
) -> tuple[dict, list[bool]]:
    """Stage 3: VLM verifier with LoRA adapter."""
    return _run_vlm_stage(
        stage_n=3,
        stage_title=f"VLM + LoRA adapter ({Path(lora_adapter_path).name})",
        pairs=pairs,
        hitl_threshold=hitl_threshold,
        batch_size=batch_size,
        lora_adapter_path=lora_adapter_path,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReID snowball loop: baseline → VLM → LoRA."
    )
    parser.add_argument("--data-root", default="/home/kokyungmin/data/WB_WoB-ReID")
    parser.add_argument("--eval-pairs", default="data/eval_pairs.jsonl")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--split", default="both_large",
                        choices=["both_large", "both_small", "with_bag", "without_bag"],
                        help="Dataset split to evaluate (default: both_large)")
    parser.add_argument("--reid-model", default="arcface-dinov2",
                        choices=["arcface-dinov2", "siglip2"])
    parser.add_argument("--reid-threshold", type=float, default=0.5,
                        help="Fixed cosine similarity threshold for Stage 1 (default: 0.5)")
    parser.add_argument("--hitl-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of pairs per VLM forward pass (default: 4)")
    parser.add_argument("--lora-adapter", default="models/vlm_verifier_lora/latest")
    parser.add_argument("--skip-stage", default="",
                        help="Comma-separated stage numbers to skip, e.g. '3'")
    parser.add_argument("--output", default="experiments/results/snowball.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    skip = {int(s) for s in args.skip_stage.split(",") if s.strip().isdigit()}

    # ── Build or load fixed eval set ─────────────────────────────────────────
    from src.evaluation.eval_dataset import build_eval_pairs, save_eval_pairs, load_eval_pairs

    eval_path = Path(args.eval_pairs)
    if eval_path.exists():
        print(f"Loading existing eval pairs: {eval_path}")
        pairs = load_eval_pairs(str(eval_path))
    else:
        print(f"Building eval pairs (n_queries={args.n_queries}, seed={args.seed}) ...")
        pairs = build_eval_pairs(
            data_root=args.data_root,
            n_queries=args.n_queries,
            seed=args.seed,
            split=args.split,
        )
        save_eval_pairs(pairs, str(eval_path))

    n_pos = sum(1 for p in pairs if p.label)
    n_neg = sum(1 for p in pairs if not p.label)
    print(f"\nEval set : {len(pairs)} pairs  (positive={n_pos}, negative={n_neg})")
    print(f"Gallery  : bounding_box_test only  [train images never used in evaluation]")
    print(SEP_WIDE)

    # ── Run stages ───────────────────────────────────────────────────────────
    stage_results: list[dict] = []

    if 1 not in skip:
        result1, _ = run_stage1_reid(pairs, args.reid_model, threshold=args.reid_threshold)
        stage_results.append(result1)

    if 2 not in skip:
        result2, _ = run_stage2_vlm(pairs, args.hitl_threshold, batch_size=args.batch_size)
        stage_results.append(result2)

    if 3 not in skip:
        lora_path = Path(args.lora_adapter)
        if not lora_path.exists():
            print(f"\n[Stage 3 SKIPPED] LoRA adapter not found: {lora_path}")
            print("  Run scripts/lora_train.py first to generate an adapter.")
        else:
            result3, _ = run_stage3_vlm_lora(pairs, str(lora_path), args.hitl_threshold, batch_size=args.batch_size)
            stage_results.append(result3)

    # ── Summary table ────────────────────────────────────────────────────────
    if stage_results:
        _print_summary_table(stage_results)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in stage_results:
        row = {k: v for k, v in r.items() if k != "metrics"}
        row["metrics"] = r["metrics"].to_dict()
        serializable.append(row)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
