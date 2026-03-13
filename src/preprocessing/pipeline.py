"""Full align pipeline orchestration.

Runs three steps in sequence:
  1. detect.py   — YOLO person detect + crop
  2. body_orientation.py — MEBOW body orientation prediction
  3. align_angle.py      — Qwen-Image-Edit angle alignment

Usage example:
    from src.preprocessing.pipeline import AlignPipeline, AlignPipelineConfig
    import cv2

    cfg = AlignPipelineConfig(detect_enabled=False, target_angle=0)
    pipeline = AlignPipeline(cfg)
    results = pipeline.process(cv2.imread("sample.jpg"))
"""
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.preprocessing.align_angle import align_to_angle, load_qwen_angle_model
from src.preprocessing.body_orientation import load_mebow_model, predict_orientation
from src.preprocessing.detect import detect_and_crop


@dataclass
class AlignPipelineConfig:
    """Configuration for AlignPipeline.

    Attributes:
        detect_enabled: If True, run YOLO person detection. If False, treat input as
            already-cropped person image.
        yolo_model_path: Path to the YOLO model weights file.
        mebow_root: Path to the cloned MEBOW repository.
        mebow_cfg_path: Path to MEBOW YAML config. None uses the default inside mebow_root.
        mebow_model_path: Path to MEBOW weights. None uses cfg.TEST.MODEL_FILE.
        target_angle: Desired output angle in degrees (0~355). Defaults to 0 (front-facing).
        device: Torch device string. None auto-detects.
    """

    detect_enabled: bool = True
    yolo_model_path: str = "./models/detect/yolo26n-pose.pt"
    mebow_root: str = "./third_party/MEBOW"
    mebow_cfg_path: str | None = None
    mebow_model_path: str | None = None
    target_angle: int = 0
    device: str | None = None


@dataclass
class AlignedResult:
    """Output of a single person processed by AlignPipeline.

    Attributes:
        image: Final aligned BGR image.
        current_angle: MEBOW-predicted body orientation of the input crop.
        target_angle: Target angle specified in config (or per-call override).
        bbox: Bounding box (x1, y1, x2, y2) if detection was run, else None.
    """

    image: np.ndarray
    current_angle: int
    target_angle: int
    bbox: tuple[int, int, int, int] | None


class AlignPipeline:
    """End-to-end pipeline: detect → orientation predict → angle align."""

    def __init__(self, config: AlignPipelineConfig) -> None:
        """Load all models at initialization.

        Args:
            config: Pipeline configuration.
        """
        self._config = config

        if config.detect_enabled:
            from ultralytics import YOLO  # noqa: PLC0415

            self._yolo = YOLO(config.yolo_model_path)
        else:
            self._yolo = None

        self._mebow_model, self._device = load_mebow_model(
            mebow_root=config.mebow_root,
            cfg_path=config.mebow_cfg_path,
            model_path=config.mebow_model_path,
            device=config.device,
        )

        self._qwen_pipe = load_qwen_angle_model(device=self._device)

    def process(
        self,
        source: np.ndarray | str,
        target_angle: int | None = None,
    ) -> list[AlignedResult]:
        """Run the full align pipeline on an image.

        Args:
            source: BGR image array or path to an image file.
            target_angle: Override the config's target_angle for this call.

        Returns:
            List of AlignedResult, one per detected (or single) person.
        """
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise FileNotFoundError(f"Image not found: {source}")
        else:
            image = source

        effective_target = target_angle if target_angle is not None else self._config.target_angle

        detect_result = detect_and_crop(
            image=image,
            model=self._yolo,
            enabled=self._config.detect_enabled,
        )

        results: list[AlignedResult] = []
        for crop, box in zip(
            detect_result.crops,
            detect_result.boxes if detect_result.boxes else [None] * len(detect_result.crops),
        ):
            current_angle = predict_orientation(self._mebow_model, crop, self._device)
            aligned = align_to_angle(self._qwen_pipe, crop, current_angle, effective_target)
            results.append(
                AlignedResult(
                    image=aligned,
                    current_angle=current_angle,
                    target_angle=effective_target,
                    bbox=box,
                )
            )

        return results
