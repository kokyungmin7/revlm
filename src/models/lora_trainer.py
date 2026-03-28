"""LoRA fine-tuning for VLM ReID verifier using human-labeled HITL data.

Wraps TRL SFTTrainer to fine-tune Qwen3-VL-8B-Instruct with PEFT LoRA
on labeled (image-pair, label) examples collected via HITL.

Requires CUDA. Typical usage on GCP L4 (24GB VRAM):
    trainer = VLMLoRATrainer("Qwen/Qwen3-VL-8B-Instruct", "models/vlm_verifier_lora")
    adapter_path = trainer.train("data/hitl/labeled.jsonl")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.models.vlm_verifier import SYSTEM_PROMPT as _SYSTEM_PROMPT, USER_PROMPT as _USER_PROMPT


def _label_to_assistant_text(
    label: bool,
    confidence: float | None = None,
    reasoning: str | None = None,
) -> str:
    """Convert labeled sample to expected model output format.

    Uses the original VLM's confidence and reasoning when available,
    so the model learns calibrated confidence rather than always-high values.

    Args:
        label: True for same person, False for different.
        confidence: Original VLM confidence. Falls back to 0.95 if None.
        reasoning: Original VLM reasoning. Falls back to a template if None.

    Returns:
        Structured assistant response string.
    """
    verdict = "YES" if label else "NO"
    conf_str = f"{confidence:.2f}" if confidence is not None else "0.95"
    if reasoning is None:
        reasoning = (
            "The two crops show the same person based on consistent clothing and appearance."
            if label
            else "The two crops show different individuals based on distinct clothing and features."
        )
    return f"SAME_PERSON: {verdict}\nCONFIDENCE: {conf_str}\nREASONING: {reasoning}"


def _build_conversation(
    img_path_a: str,
    img_path_b: str,
    label: bool,
    confidence: float | None = None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Build a Qwen3-VL conversation dict for SFT training.

    Args:
        img_path_a: Absolute path to first image.
        img_path_b: Absolute path to second image.
        label: Ground truth label.
        confidence: Original VLM confidence (passed to assistant text).
        reasoning: Original VLM reasoning (passed to assistant text).

    Returns:
        Conversation dict with 'messages' key.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path_a},
                    {"type": "image", "image": img_path_b},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _label_to_assistant_text(label, confidence, reasoning)}],
            },
        ]
    }


class VLMLoRATrainer:
    """Fine-tunes Qwen3-VL-8B-Instruct with LoRA on HITL-labeled ReID pairs.

    Args:
        model_id: HuggingFace model identifier.
        output_base: Base directory for adapter checkpoints.
        device: Target device (e.g., "cuda", "cuda:0").
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        output_base: str = "models/vlm_verifier_lora",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.output_base = Path(output_base)
        self.device = device

    def _next_version_dir(self) -> Path:
        """Determine the next versioned output directory (v1, v2, ...).

        Returns:
            Path to next version directory (not yet created).
        """
        self.output_base.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            [d for d in self.output_base.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0,
        )
        next_n = int(existing[-1].name[1:]) + 1 if existing else 1
        return self.output_base / f"v{next_n}"

    def _update_latest_symlink(self, adapter_path: Path) -> None:
        """Update the 'latest' symlink to point to the new adapter.

        Args:
            adapter_path: Path to the newly saved adapter directory.
        """
        latest = self.output_base / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(adapter_path.name)

    def train(
        self,
        labeled_jsonl: str = "data/hitl/labeled.jsonl",
        min_samples: int = 100,
        num_epochs: int = 3,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        max_seq_length: int = 2048,
    ) -> str | None:
        """Fine-tune the VLM on human-labeled HITL examples.

        Args:
            labeled_jsonl: Path to labeled.jsonl from HITLCollector.
            min_samples: Skip training if fewer examples available.
            num_epochs: Training epochs.
            per_device_batch_size: Per-device batch size.
            gradient_accumulation_steps: Gradient accumulation steps.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: LoRA dropout rate.
            learning_rate: AdamW learning rate.
            max_seq_length: Maximum sequence length for tokenization.

        Returns:
            Path to saved adapter directory, or None if skipped.
        """
        # Load labeled examples
        samples: list[dict[str, Any]] = []
        labeled_path = Path(labeled_jsonl)
        if not labeled_path.exists():
            print(f"[LoRATrainer] labeled.jsonl not found: {labeled_path}")
            return None

        with open(labeled_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        n = len(samples)
        if n < min_samples:
            print(f"[LoRATrainer] Only {n} labeled samples (min={min_samples}). Skipping.")
            return None

        print(f"[LoRATrainer] Training on {n} samples...")

        # Lazy imports (heavy, CUDA-only)
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from trl import SFTConfig, SFTTrainer

        # Build conversation dataset — use original VLM confidence/reasoning when available
        conversations = [
            _build_conversation(
                s["img_path_a"],
                s["img_path_b"],
                s["label"],
                confidence=s.get("confidence"),
                reasoning=s.get("reasoning"),
            )
            for s in samples
        ]
        dataset = Dataset.from_list(conversations)

        # Load base model
        print("[LoRATrainer] Loading base model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Output directory
        output_dir = self._next_version_dir()
        output_dir.mkdir(parents=True)

        # SFT training config
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
            dataset_text_field=None,
            max_seq_length=max_seq_length,
            gradient_checkpointing=True,
        )

        # Use the full processor (not just tokenizer) so that image pixel_values
        # are computed and passed to the model during training.
        # formatting_func is intentionally omitted — SFTTrainer calls
        # processor.apply_chat_template internally, which loads images from
        # the paths stored in each message's {"type": "image", "image": path}.
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=processor,
        )

        print("[LoRATrainer] Starting training...")
        trainer.train()

        # Save adapter only (not full model)
        model.save_pretrained(str(output_dir))
        self._update_latest_symlink(output_dir)

        print(f"[LoRATrainer] Adapter saved to: {output_dir}")
        return str(output_dir)
