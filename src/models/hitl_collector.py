"""Human-In-The-Loop (HITL) data collection for VLM ReID verification.

Manages a queue of low-confidence predictions for human review and
stores labeled examples for LoRA fine-tuning.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.models.vlm_verifier import VerificationResult


@dataclass
class HITLSample:
    """A single HITL review sample.

    Attributes:
        id: Unique identifier (uuid4).
        img_path_a: Absolute path to first BGR crop (JPEG).
        img_path_b: Absolute path to second BGR crop (JPEG).
        pred_is_same: VLM prediction.
        confidence: VLM confidence in [0.0, 1.0].
        reasoning: VLM one-sentence reasoning.
        label: Human label (True=same, False=different, None=pending).
    """

    id: str
    img_path_a: str
    img_path_b: str
    pred_is_same: bool
    confidence: float
    reasoning: str
    label: bool | None = None


class HITLCollector:
    """Collects low-confidence predictions and manages CLI-based human review.

    Args:
        data_dir: Root directory for HITL data storage.
    """

    def __init__(self, data_dir: str = "data/hitl") -> None:
        self._root = Path(data_dir)
        self._images_dir = self._root / "images"
        self._queue_path = self._root / "queue.jsonl"
        self._labeled_path = self._root / "labeled.jsonl"

        self._root.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(exist_ok=True)

    def log(
        self,
        bgr_a: np.ndarray,
        bgr_b: np.ndarray,
        result: VerificationResult,
    ) -> HITLSample:
        """Save a low-confidence prediction to the review queue.

        Args:
            bgr_a: First person crop in BGR format.
            bgr_b: Second person crop in BGR format.
            result: VLM verification result.

        Returns:
            The created HITLSample.
        """
        sample_id = str(uuid.uuid4())
        path_a = str(self._images_dir / f"{sample_id}_a.jpg")
        path_b = str(self._images_dir / f"{sample_id}_b.jpg")

        cv2.imwrite(path_a, bgr_a)
        cv2.imwrite(path_b, bgr_b)

        sample = HITLSample(
            id=sample_id,
            img_path_a=path_a,
            img_path_b=path_b,
            pred_is_same=result.is_same,
            confidence=result.confidence,
            reasoning=result.reasoning,
            label=None,
        )

        with open(self._queue_path, "a") as f:
            f.write(json.dumps(asdict(sample)) + "\n")

        return sample

    def _iter_queue(self) -> Iterator[HITLSample]:
        """Yield all pending (unlabeled) samples from queue."""
        if not self._queue_path.exists():
            return
        with open(self._queue_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield HITLSample(**data)

    def _iter_labeled(self) -> Iterator[HITLSample]:
        """Yield all labeled samples."""
        if not self._labeled_path.exists():
            return
        with open(self._labeled_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield HITLSample(**data)

    def review_pending_cli(self) -> int:
        """Interactively label pending samples via CLI.

        Displays each sample's metadata and prompts for a label.
        Commands: s (same), d (different), q (quit/skip session).

        Returns:
            Number of samples labeled in this session.
        """
        pending = [s for s in self._iter_queue() if s.label is None]
        if not pending:
            print("No pending samples to review.")
            return 0

        print(f"\n=== HITL Review — {len(pending)} pending samples ===")
        print("Commands: s=same  d=different  q=quit\n")

        labeled_ids: dict[str, bool] = {}

        for i, sample in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] ID: {sample.id[:8]}...")
            print(f"  VLM prediction : {'SAME' if sample.pred_is_same else 'DIFFERENT'}")
            print(f"  Confidence     : {sample.confidence:.3f}")
            print(f"  Reasoning      : {sample.reasoning}")
            print(f"  Image A        : {sample.img_path_a}")
            print(f"  Image B        : {sample.img_path_b}")

            while True:
                cmd = input("  Label [s/d/q]: ").strip().lower()
                if cmd == "s":
                    labeled_ids[sample.id] = True
                    print("  → Labeled: SAME\n")
                    break
                elif cmd == "d":
                    labeled_ids[sample.id] = False
                    print("  → Labeled: DIFFERENT\n")
                    break
                elif cmd == "q":
                    print("\nReview session ended.")
                    self._flush_labels(labeled_ids)
                    print(f"Labeled {len(labeled_ids)} samples. Queue: {self.queue_size} remaining.")
                    return len(labeled_ids)
                else:
                    print("  Invalid command. Enter s, d, or q.")

        self._flush_labels(labeled_ids)
        print(f"\nReview complete. Labeled {len(labeled_ids)} samples.")
        print(f"Total labeled: {self.labeled_size}")
        return len(labeled_ids)

    def _flush_labels(self, labeled_ids: dict[str, bool]) -> None:
        """Move labeled samples from queue to labeled.jsonl.

        Args:
            labeled_ids: Mapping of sample id → label.
        """
        if not labeled_ids:
            return

        remaining: list[HITLSample] = []
        for sample in self._iter_queue():
            if sample.id in labeled_ids:
                sample.label = labeled_ids[sample.id]
                with open(self._labeled_path, "a") as f:
                    f.write(json.dumps(asdict(sample)) + "\n")
            else:
                remaining.append(sample)

        # Rewrite queue without labeled items
        with open(self._queue_path, "w") as f:
            for sample in remaining:
                f.write(json.dumps(asdict(sample)) + "\n")

    def log_labeled(
        self,
        bgr_a: np.ndarray,
        bgr_b: np.ndarray,
        result: VerificationResult,
        label: bool,
    ) -> HITLSample:
        """Save a prediction with a known GT label directly to labeled.jsonl.

        Used when ground truth is available from the dataset (e.g., filename-based
        person ID), so human review is not needed.  Wrong predictions are written
        straight to the training set.

        Args:
            bgr_a: First person crop in BGR format.
            bgr_b: Second person crop in BGR format.
            result: VLM verification result.
            label: Ground-truth label (True = same person).

        Returns:
            The created HITLSample (already labeled).
        """
        sample_id = str(uuid.uuid4())
        path_a = str(self._images_dir / f"{sample_id}_a.jpg")
        path_b = str(self._images_dir / f"{sample_id}_b.jpg")

        cv2.imwrite(path_a, bgr_a)
        cv2.imwrite(path_b, bgr_b)

        sample = HITLSample(
            id=sample_id,
            img_path_a=path_a,
            img_path_b=path_b,
            pred_is_same=result.is_same,
            confidence=result.confidence,
            reasoning=result.reasoning,
            label=label,
        )

        with open(self._labeled_path, "a") as f:
            f.write(json.dumps(asdict(sample)) + "\n")

        return sample

    @property
    def queue_size(self) -> int:
        """Number of pending (unlabeled) samples in the queue."""
        return sum(1 for s in self._iter_queue() if s.label is None)

    @property
    def labeled_size(self) -> int:
        """Number of human-labeled samples available for training."""
        return sum(1 for _ in self._iter_labeled())
