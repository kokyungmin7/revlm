"""Evaluation metrics for binary ReID pair verification and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class EvalMetrics:
    """Binary classification metrics for ReID pair verification.

    Attributes:
        accuracy: (TP + TN) / total.
        precision: TP / (TP + FP). 0.0 if no positive predictions.
        recall: TP / (TP + FN). 0.0 if no actual positives.
        f1: Harmonic mean of precision and recall.
        n_correct: Number of correct predictions.
        n_total: Total number of pairs evaluated.
        tp: True positives (predicted same, actually same).
        tn: True negatives (predicted different, actually different).
        fp: False positives (predicted same, actually different).
        fn: False negatives (predicted different, actually same).
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    n_correct: int
    n_total: int
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    predictions: list[bool],
    labels: list[bool],
) -> EvalMetrics:
    """Compute binary classification metrics.

    Args:
        predictions: Model predictions (True = same person).
        labels: Ground truth labels (True = same person).

    Returns:
        EvalMetrics dataclass.

    Raises:
        ValueError: If predictions and labels have different lengths.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, labels={len(labels)}"
        )

    tp = sum(p and l for p, l in zip(predictions, labels))
    tn = sum(not p and not l for p, l in zip(predictions, labels))
    fp = sum(p and not l for p, l in zip(predictions, labels))
    fn = sum(not p and l for p, l in zip(predictions, labels))

    n_total = len(predictions)
    n_correct = tp + tn
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return EvalMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        n_correct=n_correct,
        n_total=n_total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics (mAP, CMC, pairwise Rank-1).

    Attributes:
        mean_ap:        Mean Average Precision across all queries. None if retrieval not run.
        rank1:          CMC Rank-1 — fraction of queries where top-1 hit is correct. None if retrieval not run.
        rank5:          CMC Rank-5. None if retrieval not run.
        rank10:         CMC Rank-10. None if retrieval not run.
        pairwise_rank1: Fraction of queries where positive pair score > negative pair score.
                        Computed for all stages without full gallery retrieval.
        n_queries:      Number of queries evaluated.
        n_gallery:      Full gallery size (0 if retrieval not run).
    """

    mean_ap: Optional[float]
    rank1: Optional[float]
    rank5: Optional[float]
    rank10: Optional[float]
    pairwise_rank1: float
    n_queries: int
    n_gallery: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_retrieval_metrics(
    per_query_similarities: list[list[float]],
    per_query_labels: list[list[bool]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> RetrievalMetrics:
    """Compute mAP and CMC metrics for a retrieval scenario.

    For each query, gallery images are ranked by descending similarity.
    AP is computed as the mean precision at each rank where a positive is found.
    CMC@k checks whether any positive appears in the top-k ranked results.

    Args:
        per_query_similarities: Per-query cosine similarity scores, shape [n_queries][n_gallery].
        per_query_labels:        Per-query ground-truth booleans, shape [n_queries][n_gallery].
        ks:                      Rank cutoffs for CMC (default: 1, 5, 10).

    Returns:
        RetrievalMetrics with mean_ap, rank1, rank5, rank10, pairwise_rank1.

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    if len(per_query_similarities) != len(per_query_labels):
        raise ValueError(
            f"Shape mismatch: {len(per_query_similarities)} similarity rows vs "
            f"{len(per_query_labels)} label rows."
        )

    n_queries = len(per_query_similarities)
    if n_queries == 0:
        return RetrievalMetrics(
            mean_ap=0.0, rank1=0.0, rank5=0.0, rank10=0.0,
            pairwise_rank1=0.0, n_queries=0, n_gallery=0,
        )

    n_gallery = len(per_query_similarities[0])

    aps: list[float] = []
    cmc_hits: dict[int, int] = {k: 0 for k in ks}
    pw_wins = 0
    pw_total = 0

    for sims, labels in zip(per_query_similarities, per_query_labels):
        sims_arr = np.array(sims, dtype=np.float64)
        labels_arr = np.array(labels, dtype=bool)

        order = np.argsort(sims_arr)[::-1]
        sorted_labels = labels_arr[order]

        n_pos = int(sorted_labels.sum())

        # Average Precision
        if n_pos > 0:
            cumtp = np.cumsum(sorted_labels)
            ranks = np.arange(1, len(sorted_labels) + 1)
            precision_at_k = cumtp / ranks
            ap = float((precision_at_k * sorted_labels).sum() / n_pos)
        else:
            ap = 0.0
        aps.append(ap)

        # CMC@k
        for k in ks:
            k_clip = min(k, n_gallery)
            if np.any(sorted_labels[:k_clip]):
                cmc_hits[k] += 1

        # Pairwise Rank-1: positive score > negative score
        # Requires at least one positive and one negative in the gallery.
        pos_mask = labels_arr
        neg_mask = ~labels_arr
        if pos_mask.any() and neg_mask.any():
            best_pos = float(sims_arr[pos_mask].max())
            best_neg = float(sims_arr[neg_mask].max())
            if best_pos > best_neg:
                pw_wins += 1
            pw_total += 1

    mean_ap = float(np.mean(aps)) if aps else 0.0
    cmc = {k: cmc_hits[k] / n_queries for k in ks}
    pairwise_rank1 = pw_wins / pw_total if pw_total > 0 else 0.0

    return RetrievalMetrics(
        mean_ap=mean_ap,
        rank1=cmc.get(1),
        rank5=cmc.get(5),
        rank10=cmc.get(10),
        pairwise_rank1=pairwise_rank1,
        n_queries=n_queries,
        n_gallery=n_gallery,
    )


def compute_pairwise_rank1(
    pos_scores: list[float],
    neg_scores: list[float],
) -> float:
    """Fraction of queries where positive pair score exceeds negative pair score.

    Args:
        pos_scores: Similarity or confidence score for the positive pair, one per query.
        neg_scores: Similarity or confidence score for the negative pair, one per query.

    Returns:
        Pairwise Rank-1 in [0.0, 1.0].

    Raises:
        ValueError: If pos_scores and neg_scores have different lengths.
    """
    if len(pos_scores) != len(neg_scores):
        raise ValueError(
            f"Length mismatch: pos_scores={len(pos_scores)}, neg_scores={len(neg_scores)}"
        )
    if not pos_scores:
        return 0.0
    wins = sum(p > n for p, n in zip(pos_scores, neg_scores))
    return wins / len(pos_scores)


def find_best_threshold(
    similarities: list[float],
    labels: list[bool],
    n_steps: int = 101,
) -> tuple[float, EvalMetrics]:
    """Grid-search the cosine similarity threshold that maximizes accuracy.

    Args:
        similarities: Cosine similarity scores in [-1, 1].
        labels: Ground truth labels.
        n_steps: Number of threshold candidates to try.

    Returns:
        Tuple of (best_threshold, best_metrics).
    """
    best_threshold = 0.5
    best_metrics = compute_metrics([s >= 0.5 for s in similarities], labels)

    lo = min(similarities)
    hi = max(similarities)
    step = (hi - lo) / (n_steps - 1) if n_steps > 1 else 1.0

    for i in range(n_steps):
        threshold = lo + i * step
        preds = [s >= threshold for s in similarities]
        m = compute_metrics(preds, labels)
        if m.accuracy > best_metrics.accuracy:
            best_metrics = m
            best_threshold = threshold

    return best_threshold, best_metrics
