"""Visualize HITL labeled samples — what the VLM got wrong and learned from.

Reads data/hitl/labeled.jsonl and renders one PNG panel per sample showing:
  - Query (A) and Gallery (B) images side by side
  - VLM prediction vs GT label
  - Confidence score and reasoning text

Output is saved to experiments/hitl_viz/YYYYMMDD_HHMMSS/ so existing
visualizations are never overwritten.

Usage:
    uv run python scripts/visualize_hitl.py [options]

Options:
    --labeled-jsonl   Path to labeled.jsonl (default: data/hitl/labeled.jsonl)
    --output-dir      Base output directory (default: experiments/hitl_viz)
    --max-samples     Maximum panels to render (default: all)
    --img-height      Height in pixels for each image thumbnail (default: 256)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _load_rgb(path: str, target_h: int = 256) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        placeholder = np.full((target_h, max(1, target_h // 2), 3), 80, dtype=np.uint8)
        cv2.putText(placeholder, "N/A", (8, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return placeholder
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    new_w = max(1, int(w * target_h / h))
    return cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _make_panel(sample: dict, rank: int, img_height: int = 256) -> np.ndarray:
    """Render one labeled sample as an RGB numpy array.

    Layout:
      ┌─────────────────────────────────────────────────────────┐
      │  [title] GT / Pred / conf / pair rank                  │
      ├────────────────────────┬────────────────────────────────┤
      │  Image A               │  Image B                       │
      ├────────────────────────┴────────────────────────────────┤
      │  VLM reasoning                                          │
      └─────────────────────────────────────────────────────────┘
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    gt_label: bool = sample["label"]
    pred: bool = sample["pred_is_same"]
    conf: float = sample["confidence"]
    reasoning: str = sample.get("reasoning", "")

    gt_str = "SAME" if gt_label else "DIFFERENT"
    pred_str = "SAME" if pred else "DIFFERENT"
    correct = (pred == gt_label)  # should always be False for wrong predictions

    title_color = "#c0392b" if not correct else "#27ae60"
    mark = "✓" if correct else "✗"

    img_a = _load_rgb(sample["img_path_a"], img_height)
    img_b = _load_rgb(sample["img_path_b"], img_height)

    border_color_bgr = (220, 40, 40) if not correct else (0, 200, 80)
    bw = 8
    img_a = cv2.copyMakeBorder(img_a, bw, bw, bw, bw,
                                borderType=cv2.BORDER_CONSTANT, value=border_color_bgr)
    img_b = cv2.copyMakeBorder(img_b, bw, bw, bw, bw,
                                borderType=cv2.BORDER_CONSTANT, value=border_color_bgr)

    wrapped_reason = textwrap.fill(reasoning, width=100,
                                   initial_indent="Reasoning: ",
                                   subsequent_indent="           ")
    n_lines = wrapped_reason.count("\n") + 1
    ann_h = max(1.0, n_lines * 0.22 + 0.3)

    img_h_in = img_height / 96.0 + 0.4
    fig = plt.figure(figsize=(12.0, 0.55 + img_h_in + ann_h), dpi=150)
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.55, img_h_in, ann_h],
                  hspace=0.05, wspace=0.04)

    # ── Title ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    title = (
        f"{mark} Sample #{rank:04d}   "
        f"GT: {gt_str}   Pred: {pred_str}   conf={conf:.3f}\n"
        f"A: {Path(sample['img_path_a']).name}   "
        f"B: {Path(sample['img_path_b']).name}"
    )
    ax_t.text(0.5, 0.5, title, ha="center", va="center",
              fontsize=8.5, fontweight="bold", color="white",
              transform=ax_t.transAxes, multialignment="center",
              bbox=dict(boxstyle="round,pad=0.4",
                        facecolor=title_color, alpha=0.92))

    # ── Images ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.imshow(img_a)
    ax_a.set_title("Image A (query)", fontsize=7.5, pad=3)
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.imshow(img_b)
    ax_b.set_title("Image B (gallery)", fontsize=7.5, pad=3)
    ax_b.axis("off")

    # ── Annotation ────────────────────────────────────────────
    ax_ann = fig.add_subplot(gs[2, :])
    ax_ann.axis("off")
    ax_ann.text(0.01, 0.97, wrapped_reason,
                ha="left", va="top", fontsize=7.5,
                fontfamily="monospace", transform=ax_ann.transAxes,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#f5f5f5", alpha=0.9))

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    panel = buf.reshape(h_px, w_px, 4)[:, :, :3].copy()
    plt.close(fig)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize HITL labeled samples (VLM wrong predictions)."
    )
    parser.add_argument("--labeled-jsonl", default="data/hitl/labeled.jsonl")
    parser.add_argument("--output-dir", default="experiments/hitl_viz")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max panels to render (0 = all)")
    parser.add_argument("--img-height", type=int, default=256)
    args = parser.parse_args()

    labeled_path = Path(args.labeled_jsonl)
    if not labeled_path.exists():
        print(f"ERROR: {labeled_path} not found.")
        print("  Run scripts/run_hitl_inference.py first to populate labeled.jsonl.")
        return

    samples: list[dict] = []
    with open(labeled_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print("labeled.jsonl is empty — nothing to visualize.")
        return

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    n_wrong = sum(1 for s in samples if s.get("label") is not None and s["pred_is_same"] != s["label"])
    n_correct = len(samples) - n_wrong

    print(f"Samples      : {len(samples)}  (wrong={n_wrong}, correct={n_correct})")
    print(f"Output dir   : {out_dir}")
    print(f"Rendering panels ...")

    for rank, sample in enumerate(samples, 1):
        panel = _make_panel(sample, rank, args.img_height)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"sample_{rank:04d}.png"
        cv2.imwrite(str(fname), panel_bgr)
        if rank % 10 == 0 or rank == len(samples):
            print(f"  [{rank}/{len(samples)}]", flush=True)

    # ── Summary JSON ──────────────────────────────────────────
    summary = {
        "timestamp": ts,
        "labeled_jsonl": str(labeled_path),
        "n_total": len(samples),
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "avg_confidence": round(
            sum(s["confidence"] for s in samples) / len(samples), 4
        ),
        "wrong_rate": round(n_wrong / len(samples), 4) if samples else 0.0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone → {out_dir}/")
    print(f"  {len(samples)} panels + summary.json")
    print(f"  Wrong rate : {summary['wrong_rate']:.1%}  "
          f"Avg confidence : {summary['avg_confidence']:.3f}")


if __name__ == "__main__":
    main()
