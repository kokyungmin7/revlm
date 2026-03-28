"""Evaluate the ReID accuracy snowball loop across three pipeline stages.

Outputs a comparison table showing progressive accuracy improvement:
  Stage 1 — ReID backbone (cosine similarity, auto-threshold)
  Stage 2 — VLM verifier (base Qwen3-VL-8B-Instruct)
  Stage 3 — VLM verifier + HITL LoRA fine-tuned adapter

For each run a timestamped directory is created under experiments/results/
so existing results are never overwritten.  The directory contains:
  run_config.json          — CLI arguments + run timestamp
  metrics_summary.json     — per-stage aggregated metrics
  per_pair_details.json    — per-pair predictions/confidence/reasoning for all stages
  reid_miss_vlm_hit/       — panels: Stage 1 wrong, Stage 2 correct
  vlm_miss_lora_hit/       — panels: Stage 2 wrong, Stage 3 correct
  reid_hit_vlm_miss/       — panels: Stage 1 correct, Stage 2 wrong  (regression)
  vlm_hit_lora_miss/       — panels: Stage 2 correct, Stage 3 wrong  (regression)

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
    --no-viz          Disable visualization panel generation
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Per-pair detail record ─────────────────────────────────────────────────────

@dataclass
class PairDetail:
    """Per-pair prediction detail for one evaluation stage.

    Attributes:
        pair_idx:    0-based index in the eval set.
        stage:       Stage number (1, 2, or 3).
        img_path_a:  Query image path.
        img_path_b:  Gallery image path.
        person_id_a: Person ID of query.
        person_id_b: Person ID of gallery.
        label:       Ground-truth label (True = same person).
        prediction:  Model prediction (True = same person).
        confidence:  Similarity or VLM confidence in [0, 1].
        reasoning:   VLM chain-of-thought (None for Stage 1).
        correct:     prediction == label.
    """

    pair_idx: int
    stage: int
    img_path_a: str
    img_path_b: str
    person_id_a: str
    person_id_b: str
    label: bool
    prediction: bool
    confidence: Optional[float]
    reasoning: Optional[str]
    correct: bool

    def to_dict(self) -> dict:
        return asdict(self)


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
    th_str = f"  threshold={threshold:.4f}" if threshold is not None else ""
    print(f"  Accuracy  : {_fmt_pct(m.accuracy)}  ({m.n_correct}/{m.n_total}){th_str}")
    print(f"  Precision : {_fmt_pct(m.precision)}   Recall: {_fmt_pct(m.recall)}   F1: {_fmt_pct(m.f1)}")
    print(f"  TP={m.tp}  TN={m.tn}  FP={m.fp}  FN={m.fn}")


def _print_summary_table(
    stages: list[dict],
    retrieval_result: dict | None = None,
) -> None:
    print(f"\n{SEP_WIDE}")
    print("  ReID Accuracy Snowball Loop — Summary")
    print(SEP_WIDE)

    # ── Verification metrics (all stages) ─────────────────────────
    print("  Verification metrics (all stages):")
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

    # ── Pairwise Rank-1 (all stages) ──────────────────────────────
    if any("pairwise_rank1" in s for s in stages):
        print(f"\n  Pairwise Rank-1 (all stages)  "
              "[positive pair score > negative pair score]:")
        print(f"  {'Stage':<38} {'PR@1':>7}")
        print(SEP_THIN)
        for s in stages:
            if "pairwise_rank1" in s:
                print(
                    f"  {s['label']:<38} "
                    f"{_fmt_pct(s['pairwise_rank1'])}"
                )
        print(SEP_THIN)

    # ── Retrieval metrics (Stage 1 only) ──────────────────────────
    if retrieval_result is not None:
        rm = retrieval_result["retrieval_metrics"]
        print(f"\n  Retrieval metrics (Stage 1 — full gallery ranking):")
        print(SEP_THIN)
        print(f"  mAP     : {_fmt_pct(rm.mean_ap)}")
        print(f"  Rank-1  : {_fmt_pct(rm.rank1)}")
        print(f"  Rank-5  : {_fmt_pct(rm.rank5)}")
        print(f"  Rank-10 : {_fmt_pct(rm.rank10)}")
        print(f"  Queries : {rm.n_queries}   Gallery : {rm.n_gallery} images")
        print(SEP_THIN)

    print(SEP_WIDE)


# ── Visualization ──────────────────────────────────────────────────────────────

def _load_rgb(path: str, target_h: int = 256) -> np.ndarray:
    """Load image, convert BGR→RGB, resize to fixed height."""
    bgr = cv2.imread(path)
    if bgr is None:
        placeholder = np.full((target_h, target_h // 2, 3), 128, dtype=np.uint8)
        return placeholder
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    new_w = max(1, int(w * target_h / h))
    return cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _add_border(img: np.ndarray, correct: bool, width: int = 8) -> np.ndarray:
    """Add colored border: green if correct, red if wrong."""
    color = (0, 200, 80) if correct else (220, 40, 40)
    return cv2.copyMakeBorder(
        img, width, width, width, width,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


def _verdict_str(pred: bool, label: bool, conf: float | None) -> str:
    mark = "✓" if pred == label else "✗"
    verdict = "SAME" if pred else "DIFF"
    conf_s = f" (conf={conf:.3f})" if conf is not None else ""
    return f"{mark} {verdict}{conf_s}"


def _make_transition_panel(
    category_label: str,
    pair_idx: int,
    details_by_stage: dict[int, PairDetail],
    img_height: int = 256,
) -> "np.ndarray":
    """Render a single transition-case panel as an RGB numpy array.

    Layout:
      ┌────────────────────────────────────────────────────────────┐
      │  [title]  category | GT label | pair index                │
      ├──────────────────────┬─────────────────────────────────────┤
      │  Query (A)           │  Gallery (B)                        │
      │  [image with border] │  [image with border]                │
      │  (id=..., file=...)  │  (id=..., file=...)                │
      ├──────────────────────┴─────────────────────────────────────┤
      │  Stage 1 (ReID):    ✓/✗ SAME/DIFF  conf=...              │
      │  Stage 2 (VLM base): ✓/✗ SAME/DIFF  conf=...             │
      │    reason: ...                                              │
      │  Stage 3 (VLM+LoRA): ✓/✗ SAME/DIFF  conf=...            │
      │    reason: ...                                              │
      └────────────────────────────────────────────────────────────┘
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    # Pick any stage to get pair metadata
    any_detail = next(iter(details_by_stage.values()))
    label = any_detail.label
    gt_str = "SAME PERSON" if label else "DIFFERENT PERSON"

    # Category colour mapping
    category_colors = {
        "reid_miss_vlm_hit":  "#2ecc71",   # green
        "vlm_miss_lora_hit":  "#3498db",   # blue
        "reid_hit_vlm_miss":  "#e74c3c",   # red (regression)
        "vlm_hit_lora_miss":  "#e67e22",   # orange (regression)
    }
    title_color = category_colors.get(category_label, "#555555")

    category_display = {
        "reid_miss_vlm_hit":  "ReID miss → VLM hit",
        "vlm_miss_lora_hit":  "VLM miss → LoRA hit",
        "reid_hit_vlm_miss":  "ReID hit → VLM miss  (regression)",
        "vlm_hit_lora_miss":  "VLM hit → LoRA miss  (regression)",
    }
    cat_str = category_display.get(category_label, category_label)

    # ── Load and annotate images ──────────────────────────────────
    stage_nums = sorted(details_by_stage.keys())
    # Use stage 1 border if available, else earliest available stage
    stage_for_border_a = stage_nums[0]
    d_a = details_by_stage[stage_for_border_a]

    img_a = _load_rgb(any_detail.img_path_a, target_h=img_height)
    img_b = _load_rgb(any_detail.img_path_b, target_h=img_height)

    # For border correctness: stage 1 vs stage 2 (the transition pair)
    if len(stage_nums) >= 2:
        d_s1 = details_by_stage[stage_nums[0]]
        d_s2 = details_by_stage[stage_nums[1]]
    else:
        d_s1 = d_s2 = details_by_stage[stage_nums[0]]

    # Border on image: left=stage1 border, right=stage2 border
    img_a_s1 = _add_border(img_a, d_s1.correct)
    img_a_s2 = _add_border(img_a, d_s2.correct)

    # ── Build annotation text ─────────────────────────────────────
    stage_labels = {1: "Stage 1 — ReID", 2: "Stage 2 — VLM base", 3: "Stage 3 — VLM+LoRA"}
    annotation_lines: list[str] = []
    for sn in stage_nums:
        d = details_by_stage[sn]
        slabel = stage_labels.get(sn, f"Stage {sn}")
        verdict = _verdict_str(d.prediction, d.label, d.confidence)
        annotation_lines.append(f"{slabel}:  {verdict}")
        if d.reasoning:
            wrapped = textwrap.fill(d.reasoning, width=90, initial_indent="    → ", subsequent_indent="      ")
            annotation_lines.append(wrapped)
    annotation_text = "\n".join(annotation_lines)

    # ── Matplotlib layout ─────────────────────────────────────────
    n_annot_lines = annotation_text.count("\n") + 1
    annot_height_in = max(1.2, n_annot_lines * 0.22)
    fig_w = 12.0
    img_h_in = img_height / 96.0 + 0.5
    fig_h = 0.6 + img_h_in + annot_height_in + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.55, img_h_in, annot_height_in],
        hspace=0.05,
        wspace=0.04,
    )

    # Title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    title_text = (
        f"[{cat_str}]   GT: {gt_str}   |   pair #{pair_idx:04d}"
        f"\nA: {Path(any_detail.img_path_a).name}  (id={any_detail.person_id_a})   "
        f"B: {Path(any_detail.img_path_b).name}  (id={any_detail.person_id_b})"
    )
    ax_title.text(
        0.5, 0.5, title_text,
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color="white",
        transform=ax_title.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=title_color, alpha=0.9),
        multialignment="center",
    )

    # Image A — left panel: show Stage 1 (or earliest) border
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.imshow(img_a_s1)
    ax_a.set_title(
        f"Query (A)\nid={any_detail.person_id_a}",
        fontsize=7.5, pad=3,
    )
    ax_a.axis("off")

    # Image B — right panel: show Stage 2 (or latest) border
    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.imshow(img_a_s2)
    # Actually show img_b not img_a again
    ax_b.imshow(img_b)
    ax_b.set_title(
        f"Gallery (B)\nid={any_detail.person_id_b}",
        fontsize=7.5, pad=3,
    )
    ax_b.axis("off")

    # Add correctness border patches as axis spine colour
    for ax, correct in [(ax_a, d_s1.correct), (ax_b, d_s2.correct)]:
        edge_color = "#2ecc71" if correct else "#e74c3c"
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Annotation row
    ax_ann = fig.add_subplot(gs[2, :])
    ax_ann.axis("off")
    ax_ann.text(
        0.01, 0.98, annotation_text,
        ha="left", va="top",
        fontsize=7.5,
        fontfamily="monospace",
        transform=ax_ann.transAxes,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.9),
    )

    # Legend
    handles = [
        mpatches.Patch(color="#2ecc71", label="Correct prediction"),
        mpatches.Patch(color="#e74c3c", label="Wrong prediction"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 1])

    # ── Render to numpy array ─────────────────────────────────────
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    panel_rgba = buf.reshape(h_px, w_px, 4)
    panel_rgb = panel_rgba[:, :, :3].copy()
    plt.close(fig)
    return panel_rgb


def _save_transition_panels(
    category: str,
    cases: list[tuple[int, dict[int, PairDetail]]],
    out_dir: Path,
) -> None:
    """Render and save one PNG panel per transition case."""
    if not cases:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {len(cases)} '{category}' panels → {out_dir}/")
    for rank, (pair_idx, details_by_stage) in enumerate(cases, 1):
        panel = _make_transition_panel(category, pair_idx, details_by_stage)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"pair_{pair_idx:04d}_rank{rank:03d}.png"
        cv2.imwrite(str(fname), panel_bgr)


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage1_reid(
    pairs, reid_model_name: str, threshold: float = 0.5
) -> tuple[dict, list[bool], list[PairDetail]]:
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
    valid_pairs = []

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
        valid_pairs.append((i - 1, pair))  # (0-based pair_idx, pair)

    predictions = [s >= threshold for s in similarities]
    metrics = compute_metrics(predictions, labels)

    elapsed = time.time() - t0
    print(f"  Processed {len(valid_pairs)} pairs in {elapsed:.1f}s")
    _print_metrics(metrics, threshold=threshold)

    details: list[PairDetail] = []
    for (pair_idx, pair), pred, sim in zip(valid_pairs, predictions, similarities):
        _print_pair_result(
            pair_idx + 1, len(pairs),
            pair.img_path_a, pair.img_path_b,
            pair.person_id_a, pair.person_id_b,
            pair.label, pred,
            confidence=sim,
        )
        details.append(PairDetail(
            pair_idx=pair_idx,
            stage=1,
            img_path_a=pair.img_path_a,
            img_path_b=pair.img_path_b,
            person_id_a=pair.person_id_a,
            person_id_b=pair.person_id_b,
            label=pair.label,
            prediction=pred,
            confidence=float(sim),
            reasoning=None,
            correct=(pred == pair.label),
        ))

    return {
        "stage": 1,
        "label": f"1. ReID only ({reid_model_name})",
        "model": reid_model_name,
        "threshold": threshold,
        "metrics": metrics,
        "elapsed_s": elapsed,
    }, predictions, details


def run_stage1_reid_retrieval(
    queries,  # list[RetrievalQuery]
    reid_model_name: str,
) -> tuple[dict, "RetrievalMetrics"]:
    """Stage 1 full gallery retrieval evaluation.

    Embeds the entire test gallery once, then for each query computes cosine
    similarities via a matrix-vector multiply and ranks gallery images.
    Reports mAP, Rank-1/5/10, and pairwise Rank-1.

    Args:
        queries:         List of RetrievalQuery from build_retrieval_queries.
        reid_model_name: ReID backend name.

    Returns:
        Tuple of (result_dict, RetrievalMetrics).
    """
    from src.models.reid import load_reid_model
    from src.evaluation.metrics import compute_retrieval_metrics

    _print_stage_header(1, f"ReID retrieval ({reid_model_name})")
    print(f"  Loading model: {reid_model_name} ...")
    model = load_reid_model(reid_model_name)

    if not queries:
        print("  [WARN] No queries — skipping retrieval evaluation.")
        from src.evaluation.metrics import RetrievalMetrics
        empty = RetrievalMetrics(mean_ap=0.0, rank1=0.0, rank5=0.0, rank10=0.0,
                                 pairwise_rank1=0.0, n_queries=0, n_gallery=0)
        return {"stage": 1, "label": "1. ReID retrieval", "retrieval_metrics": empty,
                "elapsed_s": 0.0}, empty

    gallery_paths = queries[0].gallery_paths
    n_gallery = len(gallery_paths)
    print(f"  Embedding {n_gallery} gallery images ...")

    t0 = time.time()

    # ── Embed gallery (once, shared across all queries) ───────────
    gallery_embs: list[np.ndarray] = []
    for gi, gpath in enumerate(gallery_paths):
        bgr = cv2.imread(gpath)
        if bgr is None:
            gallery_embs.append(np.zeros(model.embed_dim, dtype=np.float32))
        else:
            gallery_embs.append(model.extract_embedding(bgr))
        if (gi + 1) % 100 == 0 or (gi + 1) == n_gallery:
            print(f"    gallery [{gi + 1}/{n_gallery}]", flush=True)

    gallery_matrix = np.stack(gallery_embs)  # (n_gallery, embed_dim)

    # ── Embed queries and compute per-query similarities ──────────
    n_queries = len(queries)
    print(f"  Ranking {n_queries} queries ...")
    per_query_sims: list[list[float]] = []
    per_query_labels: list[list[bool]] = []

    for qi, q in enumerate(queries):
        bgr = cv2.imread(q.query_path)
        if bgr is None:
            print(f"  [WARN] Could not read query {q.query_path}, skipping.")
            continue
        q_emb = model.extract_embedding(bgr)
        sims = gallery_matrix @ q_emb  # (n_gallery,)
        per_query_sims.append(sims.tolist())
        per_query_labels.append(q.relevance)

        if (qi + 1) % 20 == 0 or (qi + 1) == n_queries:
            print(f"    query  [{qi + 1}/{n_queries}]", flush=True)

    retrieval_metrics = compute_retrieval_metrics(per_query_sims, per_query_labels)
    elapsed = time.time() - t0

    print(f"\n  Retrieval completed in {elapsed:.1f}s")
    print(f"  mAP     : {_fmt_pct(retrieval_metrics.mean_ap)}")
    print(f"  Rank-1  : {_fmt_pct(retrieval_metrics.rank1)}")
    print(f"  Rank-5  : {_fmt_pct(retrieval_metrics.rank5)}")
    print(f"  Rank-10 : {_fmt_pct(retrieval_metrics.rank10)}")
    print(f"  PR@1    : {_fmt_pct(retrieval_metrics.pairwise_rank1)}")

    result = {
        "stage": 1,
        "label": f"1. ReID retrieval ({reid_model_name})",
        "model": reid_model_name,
        "retrieval_metrics": retrieval_metrics,
        "elapsed_s": elapsed,
    }
    return result, retrieval_metrics


def _compute_pairwise_rank1_from_details(details: "list[PairDetail]") -> float:
    """Fraction of queries where positive pair confidence > negative pair confidence.

    Groups PairDetail records by query image path, then compares the best
    positive score against the best negative score for each query.
    Order-independent — does not assume alternating pos/neg layout.

    Args:
        details: Per-pair prediction records for one stage.

    Returns:
        Pairwise Rank-1 in [0.0, 1.0]. Returns 0.0 if no valid queries found.
    """
    by_query: dict[str, list[PairDetail]] = {}
    for d in details:
        by_query.setdefault(d.img_path_a, []).append(d)

    n_wins = 0
    n_total = 0
    for query_details in by_query.values():
        pos_confs = [d.confidence for d in query_details if d.label and d.confidence is not None]
        neg_confs = [d.confidence for d in query_details if not d.label and d.confidence is not None]
        if pos_confs and neg_confs:
            if max(pos_confs) > max(neg_confs):
                n_wins += 1
            n_total += 1

    return n_wins / n_total if n_total > 0 else 0.0


def _run_vlm_stage(
    stage_n: int,
    stage_title: str,
    pairs,
    hitl_threshold: float,
    batch_size: int,
    lora_adapter_path: str | None = None,
) -> tuple[dict, list[bool], list[PairDetail]]:
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
    valid_pairs: list = []
    valid_pair_indices: list[int] = []
    valid_bgr: list = []
    for orig_idx, pair in enumerate(pairs):
        bgr_a = cv2.imread(pair.img_path_a)
        bgr_b = cv2.imread(pair.img_path_b)
        if bgr_a is None or bgr_b is None:
            print(f"  [WARN] Could not read images for pair ({pair.img_path_a}), skipping.")
            continue
        valid_pairs.append(pair)
        valid_pair_indices.append(orig_idx)
        valid_bgr.append((bgr_a, bgr_b))

    predictions: list[bool] = []
    results_all: list = []
    n_queued = 0
    local_idx = 0

    t0 = time.time()
    for batch_start in range(0, len(valid_bgr), batch_size):
        batch_bgr = valid_bgr[batch_start: batch_start + batch_size]
        batch_end = batch_start + len(batch_bgr)
        print(f"  Inferring [{batch_end}/{len(valid_bgr)}] ...", flush=True)

        batch_results = verifier.verify_batch(batch_bgr)
        results_all.extend(batch_results)

        for r in batch_results:
            predictions.append(r.is_same)
            pair = valid_pairs[local_idx]
            queued = r.confidence < hitl_threshold
            if queued:
                n_queued += 1
            _print_pair_result(
                local_idx + 1, len(valid_pairs),
                pair.img_path_a, pair.img_path_b,
                pair.person_id_a, pair.person_id_b,
                pair.label, r.is_same,
                confidence=r.confidence,
                reasoning=r.reasoning,
                queued=queued,
            )
            local_idx += 1

    labels = [p.label for p in valid_pairs]

    metrics = compute_metrics(predictions, labels)
    elapsed = time.time() - t0
    print(f"\n  Processed {len(valid_pairs)} pairs in {elapsed:.1f}s  (batch_size={batch_size})")
    if stage_n == 2:
        print(f"  Would queue for HITL : {n_queued} pairs (confidence < {hitl_threshold})")
    _print_metrics(metrics)

    details: list[PairDetail] = []
    for (pair, orig_idx, r) in zip(valid_pairs, valid_pair_indices, results_all):
        details.append(PairDetail(
            pair_idx=orig_idx,
            stage=stage_n,
            img_path_a=pair.img_path_a,
            img_path_b=pair.img_path_b,
            person_id_a=pair.person_id_a,
            person_id_b=pair.person_id_b,
            label=pair.label,
            prediction=r.is_same,
            confidence=float(r.confidence),
            reasoning=r.reasoning,
            correct=(r.is_same == pair.label),
        ))

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
    return row, predictions, details


def run_stage2_vlm(
    pairs, hitl_threshold: float, batch_size: int = 1
) -> tuple[dict, list[bool], list[PairDetail]]:
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
) -> tuple[dict, list[bool], list[PairDetail]]:
    """Stage 3: VLM verifier with LoRA adapter."""
    return _run_vlm_stage(
        stage_n=3,
        stage_title=f"VLM + LoRA adapter ({Path(lora_adapter_path).name})",
        pairs=pairs,
        hitl_threshold=hitl_threshold,
        batch_size=batch_size,
        lora_adapter_path=lora_adapter_path,
    )


# ── Run artifact saving ────────────────────────────────────────────────────────

def _collect_transition_cases(
    all_details: dict[int, list[PairDetail]],
) -> dict[str, list[tuple[int, dict[int, PairDetail]]]]:
    """Find transition pairs across stages.

    Returns a dict with keys:
      reid_miss_vlm_hit   — Stage 1 wrong, Stage 2 correct
      vlm_miss_lora_hit   — Stage 2 wrong, Stage 3 correct
      reid_hit_vlm_miss   — Stage 1 correct, Stage 2 wrong (regression)
      vlm_hit_lora_miss   — Stage 2 correct, Stage 3 wrong (regression)

    Each value is a list of (pair_idx, {stage_n: PairDetail}).
    """
    # Index by pair_idx → stage → PairDetail
    by_pair: dict[int, dict[int, PairDetail]] = {}
    for stage_n, details in all_details.items():
        for d in details:
            by_pair.setdefault(d.pair_idx, {})[stage_n] = d

    categories: dict[str, list] = {
        "reid_miss_vlm_hit": [],
        "vlm_miss_lora_hit": [],
        "reid_hit_vlm_miss": [],
        "vlm_hit_lora_miss": [],
    }

    for pair_idx, by_stage in sorted(by_pair.items()):
        s1 = by_stage.get(1)
        s2 = by_stage.get(2)
        s3 = by_stage.get(3)

        if s1 and s2:
            if not s1.correct and s2.correct:
                categories["reid_miss_vlm_hit"].append((pair_idx, {1: s1, 2: s2}))
            elif s1.correct and not s2.correct:
                categories["reid_hit_vlm_miss"].append((pair_idx, {1: s1, 2: s2}))

        if s2 and s3:
            if not s2.correct and s3.correct:
                categories["vlm_miss_lora_hit"].append((pair_idx, {2: s2, 3: s3}))
            elif s2.correct and not s3.correct:
                categories["vlm_hit_lora_miss"].append((pair_idx, {2: s2, 3: s3}))

    return categories


def save_run_artifacts(
    run_dir: Path,
    args_dict: dict,
    stage_results: list[dict],
    all_details: dict[int, list[PairDetail]],
    retrieval_result: dict | None = None,
    generate_viz: bool = True,
) -> None:
    """Persist all run artifacts to a timestamped directory.

    Args:
        run_dir:          Timestamped output directory.
        args_dict:        CLI arguments as a plain dict.
        stage_results:    List of stage result dicts (with metrics).
        all_details:      {stage_n: list[PairDetail]} for all executed stages.
        retrieval_result: Stage 1 retrieval result dict, or None if not run.
        generate_viz:     Whether to render and save visualization panels.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving run artifacts → {run_dir}/")

    # ── run_config.json ───────────────────────────────────────────
    config_path = run_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"  run_config.json")

    # ── metrics_summary.json ──────────────────────────────────────
    summary = []
    for r in stage_results:
        row = {k: v for k, v in r.items() if k not in ("metrics", "retrieval_metrics")}
        if "metrics" in r:
            row["metrics"] = r["metrics"].to_dict()
        if "retrieval_metrics" in r:
            row["retrieval_metrics"] = r["retrieval_metrics"].to_dict()
        if "pairwise_rank1" in r:
            row["pairwise_rank1"] = r["pairwise_rank1"]
        summary.append(row)
    if retrieval_result is not None:
        ret_row = {k: v for k, v in retrieval_result.items() if k != "retrieval_metrics"}
        ret_row["retrieval_metrics"] = retrieval_result["retrieval_metrics"].to_dict()
        summary.append(ret_row)
    with open(run_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  metrics_summary.json")

    # ── per_pair_details.json ─────────────────────────────────────
    serialized_details: dict[str, list[dict]] = {}
    for stage_n, details in all_details.items():
        serialized_details[f"stage_{stage_n}"] = [d.to_dict() for d in details]
    with open(run_dir / "per_pair_details.json", "w") as f:
        json.dump(serialized_details, f, indent=2)
    print(f"  per_pair_details.json  ({sum(len(v) for v in all_details.values())} records)")

    # ── Transition case visualization ─────────────────────────────
    if not generate_viz:
        print("  (visualization skipped: --no-viz)")
        return

    transition_cases = _collect_transition_cases(all_details)
    total_cases = sum(len(v) for v in transition_cases.values())
    print(f"  Rendering {total_cases} transition panels ...")

    counts: dict[str, int] = {}
    for category, cases in transition_cases.items():
        counts[category] = len(cases)
        _save_transition_panels(category, cases, run_dir / category)

    # ── transition_summary.json ───────────────────────────────────
    with open(run_dir / "transition_summary.json", "w") as f:
        json.dump(counts, f, indent=2)
    print(f"  transition_summary.json  {counts}")

    print(f"\nRun complete → {run_dir}")


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
    parser.add_argument("--output", default="experiments/results/snowball.json",
                        help="Legacy flat JSON output path (still written for backward compat)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization panel generation")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Skip full gallery retrieval evaluation (mAP/Rank-k) for Stage 1")
    args = parser.parse_args()

    skip = {int(s) for s in args.skip_stage.split(",") if s.strip().isdigit()}

    # ── Timestamped run directory ─────────────────────────────────
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiments/results") / run_ts
    print(f"Run directory : {run_dir}")

    # ── Build or load fixed eval set ─────────────────────────────
    from src.evaluation.eval_dataset import (
        build_eval_pairs, save_eval_pairs, load_eval_pairs,
        build_retrieval_queries,
    )

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

    # ── Run stages ───────────────────────────────────────────────
    stage_results: list[dict] = []
    all_details: dict[int, list[PairDetail]] = {}
    retrieval_result: dict | None = None

    if 1 not in skip:
        result1, _, details1 = run_stage1_reid(pairs, args.reid_model, threshold=args.reid_threshold)
        result1["pairwise_rank1"] = _compute_pairwise_rank1_from_details(details1)
        stage_results.append(result1)
        all_details[1] = details1

        # Full gallery retrieval: mAP and Rank-k
        if not args.no_retrieval:
            print(f"\nBuilding retrieval queries (n={args.n_queries}, seed={args.seed}) ...")
            try:
                ret_queries = build_retrieval_queries(
                    data_root=args.data_root,
                    n_queries=args.n_queries,
                    seed=args.seed,
                    split=args.split,
                )
                retrieval_result, _ = run_stage1_reid_retrieval(ret_queries, args.reid_model)
            except Exception as e:
                print(f"  [WARN] Retrieval evaluation failed: {e}")
                retrieval_result = None

    if 2 not in skip:
        result2, _, details2 = run_stage2_vlm(pairs, args.hitl_threshold, batch_size=args.batch_size)
        result2["pairwise_rank1"] = _compute_pairwise_rank1_from_details(details2)
        stage_results.append(result2)
        all_details[2] = details2

    if 3 not in skip:
        lora_path = Path(args.lora_adapter)
        if not lora_path.exists():
            print(f"\n[Stage 3 SKIPPED] LoRA adapter not found: {lora_path}")
            print("  Run scripts/lora_train.py first to generate an adapter.")
        else:
            result3, _, details3 = run_stage3_vlm_lora(
                pairs, str(lora_path), args.hitl_threshold, batch_size=args.batch_size
            )
            result3["pairwise_rank1"] = _compute_pairwise_rank1_from_details(details3)
            stage_results.append(result3)
            all_details[3] = details3

    # ── Summary table ────────────────────────────────────────────
    if stage_results:
        _print_summary_table(stage_results, retrieval_result=retrieval_result)

    # ── Legacy flat output (backward compat) ─────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in stage_results:
        row = {k: v for k, v in r.items() if k not in ("metrics", "retrieval_metrics")}
        if "metrics" in r:
            row["metrics"] = r["metrics"].to_dict()
        if "pairwise_rank1" in r:
            row["pairwise_rank1"] = r["pairwise_rank1"]
        serializable.append(row)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nLegacy results saved → {out_path}")

    # ── Timestamped run artifacts ─────────────────────────────────
    args_dict = {
        "run_timestamp": run_ts,
        "data_root": args.data_root,
        "eval_pairs": args.eval_pairs,
        "n_queries": args.n_queries,
        "split": args.split,
        "reid_model": args.reid_model,
        "reid_threshold": args.reid_threshold,
        "hitl_threshold": args.hitl_threshold,
        "batch_size": args.batch_size,
        "lora_adapter": args.lora_adapter,
        "skip_stage": args.skip_stage,
        "seed": args.seed,
        "no_retrieval": args.no_retrieval,
        "n_pairs_total": len(pairs),
        "n_pairs_positive": n_pos,
        "n_pairs_negative": n_neg,
    }
    save_run_artifacts(
        run_dir=run_dir,
        args_dict=args_dict,
        stage_results=stage_results,
        all_details=all_details,
        retrieval_result=retrieval_result,
        generate_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()
