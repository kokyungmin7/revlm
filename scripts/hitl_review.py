"""CLI entry point for HITL human review of low-confidence VLM predictions.

Usage:
    uv run python scripts/hitl_review.py [--data-dir data/hitl]
"""

from __future__ import annotations

import argparse

from src.models.hitl_collector import HITLCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review low-confidence VLM ReID predictions via CLI."
    )
    parser.add_argument(
        "--data-dir",
        default="data/hitl",
        help="HITL data directory (default: data/hitl)",
    )
    args = parser.parse_args()

    collector = HITLCollector(data_dir=args.data_dir)
    print(f"HITL data directory : {args.data_dir}")
    print(f"Pending review      : {collector.queue_size}")
    print(f"Already labeled     : {collector.labeled_size}")

    if collector.queue_size == 0:
        print("\nNothing to review. Run inference with hitl_threshold set to populate the queue.")
        return

    collector.review_pending_cli()


if __name__ == "__main__":
    main()
