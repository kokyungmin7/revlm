"""Smoke tests for HITL collector — CPU compatible, no model loading required."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.models.hitl_collector import HITLCollector, HITLSample
from src.models.vlm_verifier import VerificationResult


@pytest.fixture()
def tmp_collector(tmp_path: Path) -> HITLCollector:
    """HITLCollector backed by a temporary directory."""
    return HITLCollector(data_dir=str(tmp_path / "hitl"))


@pytest.fixture()
def dummy_bgr() -> np.ndarray:
    """64x64 random BGR image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_result() -> VerificationResult:
    """Low-confidence VerificationResult for testing."""
    return VerificationResult(
        is_same=False,
        confidence=0.45,
        reasoning="Uncertain — similar clothing but different body shape.",
        raw_output="SAME_PERSON: NO\nCONFIDENCE: 0.45\nREASONING: ...",
    )


def test_hitl_collector_log_creates_files(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """Logging a sample creates image files and queue entry."""
    sample = tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    assert Path(sample.img_path_a).exists(), "Image A not saved"
    assert Path(sample.img_path_b).exists(), "Image B not saved"
    assert tmp_collector.queue_size == 1


def test_hitl_collector_log_metadata(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """Logged sample metadata matches the VerificationResult."""
    sample = tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    assert sample.pred_is_same == dummy_result.is_same
    assert sample.confidence == dummy_result.confidence
    assert sample.reasoning == dummy_result.reasoning
    assert sample.label is None


def test_hitl_queue_size_increments(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """Queue size increases with each logged sample."""
    assert tmp_collector.queue_size == 0
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)
    assert tmp_collector.queue_size == 1
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)
    assert tmp_collector.queue_size == 2


def test_hitl_labeled_size_starts_at_zero(tmp_collector: HITLCollector) -> None:
    """Labeled size is 0 before any review."""
    assert tmp_collector.labeled_size == 0


def test_hitl_review_cli_labels_sample(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """CLI review with 's' input labels the sample as same and moves to labeled."""
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    with patch("builtins.input", return_value="s"):
        count = tmp_collector.review_pending_cli()

    assert count == 1
    assert tmp_collector.queue_size == 0
    assert tmp_collector.labeled_size == 1


def test_hitl_review_cli_quit_stops_early(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """CLI review with 'q' input quits without labeling."""
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    with patch("builtins.input", return_value="q"):
        count = tmp_collector.review_pending_cli()

    assert count == 0
    assert tmp_collector.queue_size == 2
    assert tmp_collector.labeled_size == 0


def test_hitl_review_cli_different_label(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """CLI review with 'd' input labels the sample as different."""
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    with patch("builtins.input", return_value="d"):
        tmp_collector.review_pending_cli()

    labeled_path = Path(tmp_collector._labeled_path)
    record = json.loads(labeled_path.read_text().strip())
    assert record["label"] is False


def test_hitl_review_empty_queue(tmp_collector: HITLCollector) -> None:
    """Reviewing an empty queue returns 0 without error."""
    count = tmp_collector.review_pending_cli()
    assert count == 0


def test_hitl_labeled_jsonl_format(
    tmp_collector: HITLCollector,
    dummy_bgr: np.ndarray,
    dummy_result: VerificationResult,
) -> None:
    """labeled.jsonl contains valid JSON with required fields."""
    tmp_collector.log(dummy_bgr, dummy_bgr, dummy_result)

    with patch("builtins.input", return_value="s"):
        tmp_collector.review_pending_cli()

    labeled_path = Path(tmp_collector._labeled_path)
    record = json.loads(labeled_path.read_text().strip())

    required_fields = {"id", "img_path_a", "img_path_b", "pred_is_same", "confidence", "reasoning", "label"}
    assert required_fields.issubset(record.keys())
    assert record["label"] is True
