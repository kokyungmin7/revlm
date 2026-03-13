"""Clone MEBOW repository (Python source code only).
Model weights are managed separately.

Usage:
    uv run python scripts/setup_mebow.py [--target-dir PATH]
"""
import argparse
import subprocess
from pathlib import Path

MEBOW_REPO = "https://github.com/ChenyanWu/MEBOW.git"
DEFAULT_TARGET = "./third_party/MEBOW"


def clone_mebow(target_dir: str = DEFAULT_TARGET) -> None:
    """Clone the MEBOW repository to the specified directory.

    Args:
        target_dir: Destination path for the cloned repository.
    """
    target = Path(target_dir)
    if target.exists():
        print(f"Already exists: {target_dir}. Skipping.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", MEBOW_REPO, str(target)], check=True)
    print(f"Cloned to: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET,
        help=f"Destination directory (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()
    clone_mebow(args.target_dir)
