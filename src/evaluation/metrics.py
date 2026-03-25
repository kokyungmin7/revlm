"""Evaluation metrics for binary ReID pair verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
