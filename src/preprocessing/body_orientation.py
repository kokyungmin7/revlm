"""Step 2: MEBOW-based body orientation prediction.

MEBOW predicts human body orientation from a single RGB image.
Output: angle in degrees (0, 5, 10, ..., 355).

MEBOW source code must be cloned first:
    uv run python scripts/setup_mebow.py

Reference: https://github.com/ChenyanWu/MEBOW
"""
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def load_mebow_model(
    mebow_root: str = "./third_party/MEBOW",
    cfg_path: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
) -> tuple[Any, str]:
    """Load the MEBOW model for body orientation estimation.

    Args:
        mebow_root: Path to the cloned MEBOW repository root.
        cfg_path: Path to MEBOW YAML config. Defaults to the COCO config inside mebow_root.
        model_path: Path to model weights. If None, uses cfg.TEST.MODEL_FILE from config.
        device: Torch device string. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        Tuple of (model, device_string).

    Raises:
        ImportError: If MEBOW modules cannot be found at mebow_root.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add MEBOW internal modules to path
    lib_path = f"{mebow_root}/lib"
    tools_path = f"{mebow_root}/tools"
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)

    try:
        import models  # noqa: F401 — MEBOW internal
        from config import cfg, update_config  # noqa: F401 — MEBOW internal
    except ImportError as e:
        raise ImportError(
            f"Cannot import MEBOW modules from {mebow_root}. "
            "Run `uv run python scripts/setup_mebow.py` first."
        ) from e

    if cfg_path is None:
        cfg_path = f"{mebow_root}/experiments/coco/segm-4_lr1e-3.yaml"

    args = SimpleNamespace(
        cfg=cfg_path,
        opts=[],
        modelDir="",
        logDir="",
        dataDir="",
        prevModelDir="",
        device=device,
        img_path="",
    )
    update_config(cfg, args)

    if model_path is not None:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = model_path
        cfg.freeze()

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    model = model.to(device)

    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, device


def predict_orientation(
    model: Any,
    image: np.ndarray,
    device: str,
) -> int:
    """Predict body orientation angle from a BGR crop image.

    Args:
        model: Loaded MEBOW model.
        image: BGR person crop (np.ndarray, H x W x 3).
        device: Torch device string used for inference.

    Returns:
        Predicted angle in degrees: 0, 5, 10, ..., 355.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256))

    x = transform(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        _, hoe_output = model(x)
        angle = int(torch.argmax(hoe_output[0]).item() * 5)

    return angle
