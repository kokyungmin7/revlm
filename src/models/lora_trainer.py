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
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image

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
        Conversation dict with 'messages' and 'images' keys.
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
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _label_to_assistant_text(label, confidence, reasoning)}],
            },
        ],
        "images": [
            Image.open(img_path_a).convert("RGB"),
            Image.open(img_path_b).convert("RGB"),
        ],
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
        max_samples: int | None = None,
        num_epochs: int = 3,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        max_seq_length: int = 2048,
        eval_split_ratio: float = 0.1,
    ) -> str | None:
        """Fine-tune the VLM on human-labeled HITL examples.

        Args:
            labeled_jsonl: Path to labeled.jsonl from HITLCollector.
            min_samples: Skip training if fewer examples available.
            max_samples: Randomly sample up to this many examples. None = use all.
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

        if max_samples is not None and n > max_samples:
            random.shuffle(samples)
            samples = samples[:max_samples]
            print(f"[LoRATrainer] Sampled {max_samples} from {n} labeled examples.")
            n = max_samples

        # Validate image paths exist before expensive model loading
        missing = []
        for s in samples:
            for key in ("img_path_a", "img_path_b"):
                p = s.get(key, "")
                if not Path(p).exists():
                    missing.append(p)
        if missing:
            print(f"[LoRATrainer] ERROR: {len(missing)} image files not found. First 5:")
            for m in missing[:5]:
                print(f"  {m}")
            return None

        print(f"[LoRATrainer] Training on {n} samples ({n * 2} images verified)...")

        # Lazy imports (heavy, CUDA-only)
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from trl import SFTConfig, SFTTrainer

        # Build conversation dataset — use original VLM confidence/reasoning when available
        print("[LoRATrainer] Loading images into dataset...")
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

        # Log class balance — single-class data causes training collapse
        yes_count = sum(1 for s in samples if s["label"])
        no_count = n - yes_count
        print(f"[LoRATrainer] Class balance: YES={yes_count} ({yes_count/n:.1%}), NO={no_count} ({no_count/n:.1%})")
        if yes_count == 0 or no_count == 0:
            print("[LoRATrainer] WARNING: single-class dataset — training will collapse.")

        # Eval split for monitoring generalization
        eval_dataset = None
        if eval_split_ratio > 0.0 and len(dataset) >= 10:
            split = dataset.train_test_split(test_size=eval_split_ratio, seed=42)
            dataset = split["train"]
            eval_dataset = split["test"]
            print(f"[LoRATrainer] Train: {len(dataset)}, Eval: {len(eval_dataset)}")

        # Load base model with SDPA attention for padded multimodal compatibility
        print("[LoRATrainer] Loading base model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )

        # Processor with constrained image resolution for ReID crops.
        # ReID crops are small (e.g. 64x128), so we limit token count
        # to prevent OOM from dynamic resolution expansion.
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=128 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        # Apply LoRA — attention + MLP layers, vision encoder excluded
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Output directory
        output_dir = self._next_version_dir()
        output_dir.mkdir(parents=True)

        tb_log_dir = str(output_dir / "tb_logs")

        # Custom data collator: apply_chat_template + assistant-only loss masking.
        # Only computes loss on assistant response tokens (YES/NO + confidence + reasoning).
        # Masks system prompt, user message, vision tokens, and padding tokens.
        _VISION_TOKEN_IDS = [151652, 151653, 151655]
        _padding_side = processor.tokenizer.padding_side
        print(f"[LoRATrainer] Tokenizer padding_side: {_padding_side}")

        def _collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            full_texts = [
                processor.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False,
                ).strip()
                for ex in examples
            ]
            image_inputs = [ex["images"] for ex in examples]
            batch = processor(
                text=full_texts, images=image_inputs, return_tensors="pt", padding=True,
            )
            labels = batch["input_ids"].clone()

            # Mask each item: everything up to (and including) the user turn → -100
            for i, ex in enumerate(examples):
                # Tokenize only system+user to find where assistant response begins
                prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
                prompt_text = processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                ).strip()
                prompt_batch = processor(
                    text=[prompt_text], images=[ex["images"]], return_tensors="pt", padding=False,
                )
                prompt_len = prompt_batch["input_ids"].shape[1]

                # Account for left-padding: find where actual tokens start in padded row
                total_len = batch["input_ids"].shape[1]
                if _padding_side == "left":
                    seq_len = int(batch["attention_mask"][i].sum().item())
                    pad_len = total_len - seq_len
                    labels[i, : pad_len + prompt_len] = -100
                else:
                    labels[i, :prompt_len] = -100

            # Also mask vision token IDs (safety net — already covered by prompt masking,
            # but guards against any future format where vision tokens appear elsewhere)
            for tid in _VISION_TOKEN_IDS:
                labels[labels == tid] = -100

            batch["labels"] = labels
            return batch

        # SFT training config
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            max_grad_norm=0.3,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            remove_unused_columns=False,
            dataset_text_field=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="tensorboard",
            logging_dir=tb_log_dir,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=_collate_fn,
            processing_class=processor,
        )

        print("[LoRATrainer] Starting training...")
        trainer.train()

        # Save adapter only (not full model)
        model.save_pretrained(str(output_dir))
        self._update_latest_symlink(output_dir)

        print(f"[LoRATrainer] Adapter saved to: {output_dir}")
        print(f"[LoRATrainer] TensorBoard logs: {tb_log_dir}")
        print(f"[LoRATrainer] To view training curves: tensorboard --logdir {tb_log_dir}")
        return str(output_dir)
