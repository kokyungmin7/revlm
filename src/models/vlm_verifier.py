"""VLM-based ReID verification module using Qwen3-VL-8B-Instruct.

Compares two person crops and determines if they are the same individual.
Designed for GCP L4 (24GB) GPU with bfloat16 precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_SYSTEM_PROMPT = (
    "You are an expert in person re-identification. "
    "You carefully analyze visual appearance features such as clothing color, "
    "clothing style, accessories, body shape, and other distinguishing characteristics "
    "to determine if two person images show the same individual."
)

_USER_PROMPT = (
    "The two images below are person crops from surveillance cameras.\n"
    "Analyze their appearance carefully.\n"
    "Answer in exactly this format:\n"
    "SAME_PERSON: <YES or NO>\n"
    "CONFIDENCE: <0.0 to 1.0>\n"
    "REASONING: <one sentence>"
)


@dataclass
class VerificationResult:
    """Result from VLM-based person verification.

    Attributes:
        is_same: Whether the two crops are the same person.
        confidence: Confidence score in [0.0, 1.0]. 0.5 on parse failure.
        reasoning: One-sentence explanation from the model.
        raw_output: Full model output for debugging.
    """

    is_same: bool
    confidence: float
    reasoning: str
    raw_output: str


def _parse_output(raw: str) -> VerificationResult:
    """Parse structured VLM output into a VerificationResult.

    Args:
        raw: Raw text output from the model.

    Returns:
        Parsed VerificationResult. Falls back to is_same=False, confidence=0.5
        if parsing fails.
    """
    same_match = re.search(r"SAME_PERSON:\s*(YES|NO)", raw, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", raw)
    reason_match = re.search(r"REASONING:\s*(.+)", raw)

    if same_match is None:
        return VerificationResult(
            is_same=False,
            confidence=0.5,
            reasoning=raw.strip(),
            raw_output=raw,
        )

    is_same = same_match.group(1).upper() == "YES"
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = float(np.clip(confidence, 0.0, 1.0))
    reasoning = reason_match.group(1).strip() if reason_match else raw.strip()

    return VerificationResult(
        is_same=is_same,
        confidence=confidence,
        reasoning=reasoning,
        raw_output=raw,
    )


class VLMVerifier:
    """VLM-based person re-identification verifier.

    Uses Qwen3-VL-8B-Instruct to compare two person crops and determine
    if they depict the same individual.

    Args:
        device: Target device string (e.g., "cuda", "cuda:0").
        model_id: HuggingFace model identifier.
        dtype: Model precision dtype.
    """

    def __init__(
        self,
        device: str,
        model_id: str,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = dtype

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def verify(
        self,
        bgr_a: np.ndarray,
        bgr_b: np.ndarray,
        max_new_tokens: int = 256,
    ) -> VerificationResult:
        """Verify whether two BGR person crops are the same individual.

        Args:
            bgr_a: First person crop in BGR format (H, W, 3).
            bgr_b: Second person crop in BGR format (H, W, 3).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            VerificationResult with is_same, confidence, reasoning, raw_output.
        """
        pil_a = Image.fromarray(bgr_a[:, :, ::-1])
        pil_b = Image.fromarray(bgr_b[:, :, ::-1])

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_a},
                    {"type": "image", "image": pil_b},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = output_ids[:, input_len:]
        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return _parse_output(raw)


def load_vlm_verifier(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> VLMVerifier:
    """Load a VLMVerifier with the specified configuration.

    Args:
        model_id: HuggingFace model identifier for the VLM.
        device: Target device. None auto-detects CUDA.
        dtype: Model precision. Defaults to bfloat16.

    Returns:
        Loaded VLMVerifier ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return VLMVerifier(device=device, model_id=model_id, dtype=dtype)
