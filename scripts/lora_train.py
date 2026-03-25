"""CLI entry point for LoRA fine-tuning on HITL-labeled ReID pairs.

Usage:
    uv run python scripts/lora_train.py [options]

Options:
    --labeled-jsonl   Path to labeled.jsonl (default: data/hitl/labeled.jsonl)
    --output-base     Base dir for adapters (default: models/vlm_verifier_lora)
    --model-id        HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct)
    --min-samples     Minimum labeled examples required (default: 100)
    --epochs          Training epochs (default: 3)
    --lora-r          LoRA rank (default: 16)
    --lora-alpha      LoRA alpha (default: 32)
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune VLM verifier with LoRA on HITL-labeled data."
    )
    parser.add_argument("--labeled-jsonl", default="data/hitl/labeled.jsonl")
    parser.add_argument("--output-base", default="models/vlm_verifier_lora")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for LoRA training. Aborting.")
        return

    from src.models.lora_trainer import VLMLoRATrainer

    trainer = VLMLoRATrainer(
        model_id=args.model_id,
        output_base=args.output_base,
    )

    adapter_path = trainer.train(
        labeled_jsonl=args.labeled_jsonl,
        min_samples=args.min_samples,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if adapter_path:
        print(f"\nTraining complete. Adapter saved to: {adapter_path}")
        print(f"To use: load_vlm_verifier(lora_adapter_path='{adapter_path}')")
    else:
        print("\nTraining skipped (not enough labeled samples or error).")


if __name__ == "__main__":
    main()
