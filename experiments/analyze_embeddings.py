"""Analyze and visualize ReID embeddings from extracted JSON.

Usage:
    uv run python experiments/analyze_embeddings.py \\
        --embeddings experiments/outputs/embeddings.json \\
        --target_set both_large \\
        --output_dir experiments/outputs/figures

Outputs (saved to --output_dir):
    01_intra_inter_distribution.png   - Same-person vs different-person similarity KDE
    02_similarity_heatmap.png         - NxN cosine similarity matrix heatmap
    03_bagset_boxplot.png             - Intra-person similarity by bag_set
    04_angle_vs_similarity.png        - Angle difference vs similarity (same-person pairs)
    05_camera_similarity.png          - Same-camera vs diff-camera similarity
    06_tsne_embeddings.png            - t-SNE scatter plot
    07_case_examples.png              - Image pairs: hard negatives, easy/ambiguous/hard positives, large angle diff
    summary_stats.json                - Numeric summary of all analyses
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

matplotlib.use("Agg")

# ── constants ─────────────────────────────────────────────────────────────────
_INTER_SAMPLE_MAX = 100_000
_HEATMAP_MAX_PERSONS = 80
_TSNE_MAX_SAMPLES = 5_000
_ANGLE_BINS = [(0, "front"), (90, "right"), (180, "back"), (270, "left")]
_ANGLE_BIN_HALF = 45
_FIG_DPI = 150

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _angle_diff(a: int, b: int) -> int:
    """Circular difference between two angles (0-355°). Result in [0, 180]."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _angle_bin_label(angle: int) -> str:
    """Assign angle to a cardinal direction bin."""
    for center, label in _ANGLE_BINS:
        d = abs(angle - center) % 360
        if min(d, 360 - d) <= _ANGLE_BIN_HALF:
            return label
    return "other"


def _load_data(
    embeddings_path: Path,
    target_set: str | None,
) -> tuple[list[dict], np.ndarray, str | None]:
    """Load JSON and optionally filter to a single set.

    Returns:
        (records, embeddings_matrix, dataset_root) where embeddings_matrix is (N, 768).
    """
    with open(embeddings_path) as f:
        data = json.load(f)

    dataset_root: str | None = data.get("metadata", {}).get("dataset_root")

    images = data["images"]
    if target_set:
        images = [r for r in images if r["set"] == target_set]

    if not images:
        raise ValueError(
            f"No images found for set='{target_set}'. "
            f"Available sets: {list({r['set'] for r in data['images']})}"
        )

    embs = np.array([r["embedding"] for r in images], dtype=np.float32)
    return images, embs, dataset_root


def _build_pair_sims(
    records: list[dict],
    embs: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Compute intra-person and sampled inter-person cosine similarities.

    Returns:
        (intra_sims, inter_sims)
    """
    # Group indices by person_id
    from collections import defaultdict
    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    # Intra-person: all pairs within same person
    intra_sims: list[float] = []
    for idxs in person_idx.values():
        for i, j in itertools.combinations(idxs, 2):
            intra_sims.append(float(np.dot(embs[i], embs[j])))

    # Inter-person: random sampled pairs across different persons
    all_persons = list(person_idx.keys())
    inter_sims: list[float] = []
    rng = random.Random(42)
    attempts = 0
    max_attempts = _INTER_SAMPLE_MAX * 3
    while len(inter_sims) < _INTER_SAMPLE_MAX and attempts < max_attempts:
        pid_a, pid_b = rng.sample(all_persons, 2)
        i = rng.choice(person_idx[pid_a])
        j = rng.choice(person_idx[pid_b])
        inter_sims.append(float(np.dot(embs[i], embs[j])))
        attempts += 1

    return intra_sims, inter_sims


# ── image loader helper ───────────────────────────────────────────────────────

def _load_image(abs_path: Path, size: tuple[int, int] = (96, 192)) -> np.ndarray:
    """Load image as RGB numpy array, resized to (width, height).

    Returns a gray placeholder if the file is missing or unreadable.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(abs_path).convert("RGB").resize(size, PILImage.BILINEAR)
        return np.array(img)
    except Exception:
        placeholder = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
        return placeholder


# ── plot 07: case example image grid ─────────────────────────────────────────

def plot_case_examples(
    records: list[dict],
    embs: np.ndarray,
    dataset_root: Path,
    output_path: Path,
    n: int = 5,
) -> None:
    """Show image pairs for 5 interesting cases.

    Cases:
        1. Hard negatives   — different person, highest cosine similarity
        2. Easy positives   — same person, highest cosine similarity
        3. Large angle diff — same person, largest angle difference
        4. Ambiguous        — same person, similarity near median (borderline)
        5. Hard positives   — same person, lowest cosine similarity
    """
    from collections import defaultdict

    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    # ── Case 1: hard negatives (different person, high sim) ───────────────────
    # Sample a subset of persons to keep memory manageable
    rng = random.Random(42)
    all_pids = list(person_idx.keys())
    sample_pids = rng.sample(all_pids, min(2000, len(all_pids)))
    sample_idx = [person_idx[pid][0] for pid in sample_pids]  # one image per person
    sample_embs = embs[sample_idx]

    sim_mat = sample_embs @ sample_embs.T  # (M, M)
    hard_neg_pairs: list[tuple[float, int, int]] = []
    for row in range(len(sample_idx)):
        for col in range(row + 1, len(sample_idx)):
            pid_a = records[sample_idx[row]]["person_id"]
            pid_b = records[sample_idx[col]]["person_id"]
            if pid_a != pid_b:
                hard_neg_pairs.append((float(sim_mat[row, col]), sample_idx[row], sample_idx[col]))
    hard_neg_pairs.sort(reverse=True)
    hard_neg_top = hard_neg_pairs[:n]

    # ── Cases 2, 4, 5: all intra-person pairs sorted by similarity ────────────
    all_pos_pairs: list[tuple[float, int, int]] = []
    for idxs in person_idx.values():
        for i, j in itertools.combinations(idxs, 2):
            all_pos_pairs.append((float(np.dot(embs[i], embs[j])), i, j))
    all_pos_pairs.sort(reverse=True)

    # Case 2: easy positives (highest sim)
    easy_pos_top = all_pos_pairs[:n]

    # Case 5: hard positives (lowest sim)
    hard_pos_top = list(reversed(all_pos_pairs[-n:]))

    # Case 4: ambiguous (median ± window)
    mid = len(all_pos_pairs) // 2
    half_n = n // 2
    ambiguous_top = all_pos_pairs[max(0, mid - half_n): mid - half_n + n]

    # ── Case 3: large angle diff (same person) ────────────────────────────────
    large_angle_pairs: list[tuple[int, float, int, int]] = []
    for idxs in person_idx.values():
        for i, j in itertools.combinations(idxs, 2):
            d = _angle_diff(records[i]["orientation_angle"], records[j]["orientation_angle"])
            large_angle_pairs.append((d, float(np.dot(embs[i], embs[j])), i, j))
    large_angle_pairs.sort(reverse=True)
    large_angle_top = large_angle_pairs[:n]

    # ── Build figure ──────────────────────────────────────────────────────────
    case_specs = [
        ("Hard Negatives (diff person, high sim)", hard_neg_top, "sim"),
        ("Easy Positives (same person, high sim)", easy_pos_top, "sim"),
        ("Ambiguous (same person, mid sim)", ambiguous_top, "sim"),
        ("Hard Positives (same person, low sim)", hard_pos_top, "sim"),
        ("Large Angle Diff (same person)", large_angle_top, "angle"),
    ]

    img_w, img_h = 96, 192
    n_cols = n * 2          # left + right image for each pair
    n_rows = len(case_specs)
    fig_w = max(16, n_cols * 1.2)
    fig_h = n_rows * 3.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (title, pairs, metric) in enumerate(case_specs):
        # Section title on the leftmost axis
        axes[row_idx, 0].set_ylabel(title, fontsize=9, fontweight="bold", rotation=90, labelpad=6)
        for col_pair, pair in enumerate(pairs):
            if metric == "sim":
                sim_val, i, j = pair
                angle_d = _angle_diff(records[i]["orientation_angle"], records[j]["orientation_angle"])
                caption_left = (f"pid={records[i]['person_id']}\n"
                                f"cam={records[i]['camera_id']}  {records[i]['orientation_angle']}°")
                caption_right = (f"pid={records[j]['person_id']}\n"
                                 f"cam={records[j]['camera_id']}  {records[j]['orientation_angle']}°\n"
                                 f"sim={sim_val:.3f}  Δangle={angle_d}°")
            else:  # angle metric
                angle_d, sim_val, i, j = pair
                caption_left = (f"pid={records[i]['person_id']}\n"
                                f"cam={records[i]['camera_id']}  {records[i]['orientation_angle']}°")
                caption_right = (f"pid={records[j]['person_id']}\n"
                                 f"cam={records[j]['camera_id']}  {records[j]['orientation_angle']}°\n"
                                 f"Δangle={angle_d}°  sim={sim_val:.3f}")

            ax_l = axes[row_idx, col_pair * 2]
            ax_r = axes[row_idx, col_pair * 2 + 1]

            img_l = _load_image(dataset_root / records[i]["path"], (img_w, img_h))
            img_r = _load_image(dataset_root / records[j]["path"], (img_w, img_h))

            ax_l.imshow(img_l)
            ax_l.set_xlabel(caption_left, fontsize=6)
            ax_l.set_xticks([])
            ax_l.set_yticks([])

            ax_r.imshow(img_r)
            ax_r.set_xlabel(caption_right, fontsize=6)
            ax_r.set_xticks([])
            ax_r.set_yticks([])

            # Vertical separator between pairs (thin red line on left edge of right image)
            if col_pair > 0:
                ax_l.spines["left"].set_linewidth(2)
                ax_l.spines["left"].set_color("lightgray")

        # Hide unused axes if pairs < n
        for col_pair in range(len(pairs), n):
            axes[row_idx, col_pair * 2].set_visible(False)
            axes[row_idx, col_pair * 2 + 1].set_visible(False)

    fig.suptitle("Case Examples — ReID Embedding Analysis", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── plot 01: intra vs inter distribution ─────────────────────────────────────

def plot_intra_inter_distribution(
    intra: list[float],
    inter: list[float],
    output_path: Path,
) -> dict:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(intra, ax=ax, label=f"Same person (n={len(intra):,})", fill=True, alpha=0.4, color="steelblue")
    sns.kdeplot(inter, ax=ax, label=f"Diff person (n={len(inter):,})", fill=True, alpha=0.4, color="tomato")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Intra-person vs Inter-person Cosine Similarity Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)

    return {
        "intra_mean": float(np.mean(intra)),
        "intra_std": float(np.std(intra)),
        "inter_mean": float(np.mean(inter)),
        "inter_std": float(np.std(inter)),
        "separation": float(np.mean(intra) - np.mean(inter)),
    }


# ── plot 02: similarity heatmap ───────────────────────────────────────────────

def plot_similarity_heatmap(
    records: list[dict],
    embs: np.ndarray,
    output_path: Path,
) -> None:
    from collections import defaultdict
    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    # Pick first image per person, up to max persons (sorted by person_id)
    sorted_pids = sorted(person_idx.keys())[:_HEATMAP_MAX_PERSONS]
    sel_indices = [person_idx[pid][0] for pid in sorted_pids]
    sel_embs = embs[sel_indices]

    sim_matrix = sel_embs @ sel_embs.T  # (N, N) cosine similarity
    n = len(sorted_pids)

    fig, ax = plt.subplots(figsize=(max(8, n // 4), max(7, n // 4)))
    sns.heatmap(
        sim_matrix,
        ax=ax,
        vmin=-0.3,
        vmax=1.0,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title(f"Similarity Matrix ({n} persons, 1 image each)")
    ax.set_xlabel("Person index")
    ax.set_ylabel("Person index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)


# ── plot 03: bagset boxplot ───────────────────────────────────────────────────

def plot_bagset_boxplot(
    records: list[dict],
    embs: np.ndarray,
    output_path: Path,
) -> dict:
    from collections import defaultdict
    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    bag_sims: dict[str, list[float]] = {"with_bag_person": [], "without_bag_person": []}

    for pid, idxs in person_idx.items():
        if len(idxs) < 2:
            continue
        bag_set = records[idxs[0]]["bag_set"]
        for i, j in itertools.combinations(idxs, 2):
            sim = float(np.dot(embs[i], embs[j]))
            bag_sims[bag_set].append(sim)

    labels = [k for k, v in bag_sims.items() if v]
    data = [bag_sims[k] for k in labels]

    if not data:
        plt.figure()
        plt.text(0.5, 0.5, "No intra-person pairs found", ha="center")
        plt.savefig(output_path, dpi=_FIG_DPI)
        plt.close()
        return {}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"),
               medianprops=dict(color="navy", linewidth=2))
    ax.set_ylabel("Intra-person Cosine Similarity")
    ax.set_title("Intra-person Similarity by Bag Set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_pairs": len(v)}
            for k, v in bag_sims.items() if v}


# ── plot 04: angle difference vs similarity ───────────────────────────────────

def plot_angle_vs_similarity(
    records: list[dict],
    embs: np.ndarray,
    output_path: Path,
) -> dict:
    from collections import defaultdict
    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    angle_diffs: list[int] = []
    sims: list[float] = []

    for idxs in person_idx.values():
        for i, j in itertools.combinations(idxs, 2):
            d = _angle_diff(records[i]["orientation_angle"], records[j]["orientation_angle"])
            angle_diffs.append(d)
            sims.append(float(np.dot(embs[i], embs[j])))

    if not angle_diffs:
        plt.figure()
        plt.text(0.5, 0.5, "No intra-person pairs found", ha="center")
        plt.savefig(output_path, dpi=_FIG_DPI)
        plt.close()
        return {}

    # Bin by angle difference
    bins = [0, 30, 60, 90, 120, 150, 181]
    bin_labels = ["0-30°", "30-60°", "60-90°", "90-120°", "120-150°", "150-180°"]
    bin_groups: dict[str, list[float]] = {lbl: [] for lbl in bin_labels}

    for d, s in zip(angle_diffs, sims):
        for k, (lo, hi) in enumerate(zip(bins, bins[1:])):
            if lo <= d < hi:
                bin_groups[bin_labels[k]].append(s)
                break

    bin_means = [np.mean(v) if v else np.nan for v in bin_groups.values()]
    bin_stds = [np.std(v) if v else np.nan for v in bin_groups.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter (sampled)
    max_scatter = 10_000
    if len(angle_diffs) > max_scatter:
        idx = random.sample(range(len(angle_diffs)), max_scatter)
        angle_diffs_s = [angle_diffs[i] for i in idx]
        sims_s = [sims[i] for i in idx]
    else:
        angle_diffs_s, sims_s = angle_diffs, sims

    axes[0].scatter(angle_diffs_s, sims_s, alpha=0.15, s=8, color="steelblue")
    z = np.polyfit(angle_diffs, sims, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 180, 100)
    axes[0].plot(x_line, p(x_line), color="tomato", linewidth=2, label="trend")
    axes[0].set_xlabel("Angle Difference (°)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Angle Diff vs Similarity (same-person pairs)")
    axes[0].legend()

    # Bar per bin
    valid_labels = [lbl for lbl, m in zip(bin_labels, bin_means) if not np.isnan(m)]
    valid_means = [m for m in bin_means if not np.isnan(m)]
    valid_stds = [s for s, m in zip(bin_stds, bin_means) if not np.isnan(m)]

    axes[1].bar(valid_labels, valid_means, yerr=valid_stds, capsize=4,
                color="steelblue", edgecolor="navy", alpha=0.8)
    axes[1].set_xlabel("Angle Difference Bin")
    axes[1].set_ylabel("Mean Cosine Similarity")
    axes[1].set_title("Mean Similarity by Angle Difference Bin")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)

    return {
        "bins": {lbl: {"mean": float(np.mean(v)), "n_pairs": len(v)}
                 for lbl, v in bin_groups.items() if v},
        "pearson_r": float(np.corrcoef(angle_diffs, sims)[0, 1]),
    }


# ── plot 05: camera similarity ────────────────────────────────────────────────

def plot_camera_similarity(
    records: list[dict],
    embs: np.ndarray,
    output_path: Path,
) -> dict:
    from collections import defaultdict
    person_idx: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        person_idx[rec["person_id"]].append(i)

    same_cam: list[float] = []
    diff_cam: list[float] = []

    for idxs in person_idx.values():
        for i, j in itertools.combinations(idxs, 2):
            sim = float(np.dot(embs[i], embs[j]))
            if records[i]["camera_id"] == records[j]["camera_id"]:
                same_cam.append(sim)
            else:
                diff_cam.append(sim)

    if not same_cam and not diff_cam:
        plt.figure()
        plt.text(0.5, 0.5, "No data", ha="center")
        plt.savefig(output_path, dpi=_FIG_DPI)
        plt.close()
        return {}

    fig, ax = plt.subplots(figsize=(7, 5))
    data, labels = [], []
    if same_cam:
        data.append(same_cam)
        labels.append(f"Same camera\n(n={len(same_cam):,})")
    if diff_cam:
        data.append(diff_cam)
        labels.append(f"Diff camera\n(n={len(diff_cam):,})")

    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"),
               medianprops=dict(color="navy", linewidth=2))
    ax.set_ylabel("Cosine Similarity (same-person pairs)")
    ax.set_title("Same-camera vs Cross-camera Similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)

    stats = {}
    if same_cam:
        stats["same_camera"] = {"mean": float(np.mean(same_cam)), "std": float(np.std(same_cam)), "n": len(same_cam)}
    if diff_cam:
        stats["diff_camera"] = {"mean": float(np.mean(diff_cam)), "std": float(np.std(diff_cam)), "n": len(diff_cam)}
    return stats


# ── plot 06: t-SNE ────────────────────────────────────────────────────────────

def plot_tsne(
    records: list[dict],
    embs: np.ndarray,
    output_path: Path,
) -> None:
    n = len(records)
    if n > _TSNE_MAX_SAMPLES:
        # Stratified sampling by person_id
        from collections import defaultdict
        person_idx: dict[int, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            person_idx[rec["person_id"]].append(i)
        rng = random.Random(42)
        selected: list[int] = []
        per_person = max(1, _TSNE_MAX_SAMPLES // len(person_idx))
        for idxs in person_idx.values():
            selected.extend(rng.sample(idxs, min(per_person, len(idxs))))
        selected = selected[:_TSNE_MAX_SAMPLES]
        records = [records[i] for i in selected]
        embs = embs[selected]

    print(f"      Running t-SNE on {len(records)} samples ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(records) - 1))
    coords = tsne.fit_transform(embs)

    bag_sets = [r["bag_set"] for r in records]
    angle_bins = [_angle_bin_label(r["orientation_angle"]) for r in records]
    person_ids = [r["person_id"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color by bag_set
    unique_bags = sorted(set(bag_sets))
    palette_bags = sns.color_palette("Set1", len(unique_bags))
    bag_color_map = dict(zip(unique_bags, palette_bags))
    colors_bag = [bag_color_map[b] for b in bag_sets]
    axes[0].scatter(coords[:, 0], coords[:, 1], c=colors_bag, s=10, alpha=0.5)
    for label, color in bag_color_map.items():
        axes[0].scatter([], [], c=[color], label=label, s=40)
    axes[0].legend(fontsize=8)
    axes[0].set_title("t-SNE — colored by bag set")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Color by angle bin
    unique_angles = sorted(set(angle_bins))
    palette_angles = sns.color_palette("tab10", len(unique_angles))
    angle_color_map = dict(zip(unique_angles, palette_angles))
    colors_angle = [angle_color_map[b] for b in angle_bins]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=colors_angle, s=10, alpha=0.5)
    for label, color in angle_color_map.items():
        axes[1].scatter([], [], c=[color], label=label, s=40)
    axes[1].legend(fontsize=8)
    axes[1].set_title("t-SNE — colored by orientation bin")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Color by person_id (only if ≤ 30 persons, else use continuous colormap)
    unique_pids = sorted(set(person_ids))
    if len(unique_pids) <= 30:
        palette_pids = sns.color_palette("tab20", len(unique_pids))
        pid_color_map = dict(zip(unique_pids, palette_pids))
        colors_pid = [pid_color_map[p] for p in person_ids]
        axes[2].scatter(coords[:, 0], coords[:, 1], c=colors_pid, s=10, alpha=0.5)
    else:
        pid_norm = [(p - min(unique_pids)) / max(1, (max(unique_pids) - min(unique_pids)))
                    for p in person_ids]
        axes[2].scatter(coords[:, 0], coords[:, 1], c=pid_norm, cmap="hsv", s=10, alpha=0.5)
    axes[2].set_title(f"t-SNE — colored by person ID ({len(unique_pids)} persons)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize ReID embeddings from extracted JSON."
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Path to embeddings JSON produced by extract_embeddings.py",
    )
    parser.add_argument(
        "--target_set",
        default=None,
        help="Filter to a single dataset set (e.g. both_large). Default: all sets.",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/outputs/figures",
        type=Path,
        help="Directory for output figures and summary JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        type=Path,
        help="Root directory of the image dataset. Defaults to metadata.dataset_root in the JSON.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=5,
        help="Number of image pairs to show per case in plot 07 (default: 5).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Loading embeddings from {args.embeddings} ...")
    records, embs, meta_dataset_root = _load_data(args.embeddings, args.target_set)
    print(f"  {len(records)} images, set filter: {args.target_set or 'all'}")

    dataset_root: Path | None = args.dataset_root or (
        Path(meta_dataset_root) if meta_dataset_root else None
    )

    n_persons = len({r["person_id"] for r in records})
    print(f"  {n_persons} unique persons")

    # ── compute pair similarities ─────────────────────────────────────────────
    print("Computing pairwise similarities ...")
    intra_sims, inter_sims = _build_pair_sims(records, embs)
    print(f"  {len(intra_sims):,} intra-person pairs, {len(inter_sims):,} inter-person pairs")

    summary: dict = {}

    # ── plot 01 ───────────────────────────────────────────────────────────────
    print("Plotting 01: intra/inter distribution ...")
    stats = plot_intra_inter_distribution(
        intra_sims, inter_sims, out_dir / "01_intra_inter_distribution.png"
    )
    summary["intra_inter"] = stats

    # ── plot 02 ───────────────────────────────────────────────────────────────
    print("Plotting 02: similarity heatmap ...")
    plot_similarity_heatmap(records, embs, out_dir / "02_similarity_heatmap.png")

    # ── plot 03 ───────────────────────────────────────────────────────────────
    print("Plotting 03: bag set boxplot ...")
    stats = plot_bagset_boxplot(records, embs, out_dir / "03_bagset_boxplot.png")
    summary["bagset"] = stats

    # ── plot 04 ───────────────────────────────────────────────────────────────
    print("Plotting 04: angle vs similarity ...")
    stats = plot_angle_vs_similarity(records, embs, out_dir / "04_angle_vs_similarity.png")
    summary["angle_vs_sim"] = stats

    # ── plot 05 ───────────────────────────────────────────────────────────────
    print("Plotting 05: camera similarity ...")
    stats = plot_camera_similarity(records, embs, out_dir / "05_camera_similarity.png")
    summary["camera"] = stats

    # ── plot 06 ───────────────────────────────────────────────────────────────
    print("Plotting 06: t-SNE ...")
    plot_tsne(records, embs, out_dir / "06_tsne_embeddings.png")

    # ── plot 07 ───────────────────────────────────────────────────────────────
    if dataset_root is not None:
        print(f"Plotting 07: case examples (n={args.n_examples} per case) ...")
        plot_case_examples(
            records, embs, dataset_root, out_dir / "07_case_examples.png", n=args.n_examples
        )
    else:
        print("Skipping plot 07: dataset_root not available (pass --dataset_root or check JSON metadata).")

    # ── save summary ─────────────────────────────────────────────────────────
    summary_path = out_dir / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll figures saved to {out_dir}")
    print(f"Summary stats saved to {summary_path}")

    # ── print key numbers ─────────────────────────────────────────────────────
    print("\n── Key Results ──────────────────────────────────────────")
    si = summary.get("intra_inter", {})
    if si:
        print(f"  Same-person similarity:  {si['intra_mean']:.4f} ± {si['intra_std']:.4f}")
        print(f"  Diff-person similarity:  {si['inter_mean']:.4f} ± {si['inter_std']:.4f}")
        print(f"  Separation (intra-inter): {si['separation']:.4f}")
    for bset, bstats in summary.get("bagset", {}).items():
        print(f"  Intra-sim [{bset}]: {bstats['mean']:.4f} ± {bstats['std']:.4f} ({bstats['n_pairs']:,} pairs)")
    cam = summary.get("camera", {})
    if "same_camera" in cam and "diff_camera" in cam:
        print(f"  Same-camera sim:   {cam['same_camera']['mean']:.4f}")
        print(f"  Cross-camera sim:  {cam['diff_camera']['mean']:.4f}")


if __name__ == "__main__":
    main()
