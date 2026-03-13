"""Step 3: Qwen-Image-Edit based angle alignment.

Converts a person image from its current body orientation angle to a target angle
using Qwen-Image-Edit-2509 with a LoRA adapter trained for camera rotation.

Model stack:
  - Base pipeline  : Qwen/Qwen-Image-Edit-2509
  - Transformer    : linoyts/Qwen-Image-Edit-Rapid-AIO (subfolder='transformer')
  - LoRA adapter   : dx8152/Qwen-Edit-2509-Multiple-angles
"""
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


# LoRA weight filename in the HuggingFace repository.
# Verify at: https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles
_LORA_WEIGHT_NAME = "lora_camera_rotation.safetensors"


def load_qwen_angle_model(
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load Qwen-Image-Edit pipeline with LoRA for angle rotation.

    Args:
        device: Torch device string. Defaults to 'cuda' if available, else 'cpu'.
        dtype: Model weight dtype. Defaults to bfloat16.

    Returns:
        Loaded QwenImageEditPlusPipeline instance.
    """
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)
    pipe.load_lora_weights(
        "dx8152/Qwen-Edit-2509-Multiple-angles",
        weight_name=_LORA_WEIGHT_NAME,
        adapter_name="angles",
    )

    return pipe


def align_to_angle(
    pipe: Any,
    image: np.ndarray,
    current_angle: int,
    target_angle: int = 0,
) -> np.ndarray:
    """Rotate a person image from current_angle to target_angle.

    If the angular delta is less than 5 degrees, the original image is returned
    unchanged without calling the model.

    Args:
        pipe: Loaded QwenImageEditPlusPipeline.
        image: BGR person crop (np.ndarray, H x W x 3).
        current_angle: MEBOW-predicted current orientation in degrees (0~355).
        target_angle: Desired orientation in degrees (0~355). Defaults to 0 (front).

    Returns:
        BGR image aligned to target_angle.
    """
    delta = ((target_angle - current_angle) + 180) % 360 - 180  # in [-180, 180]

    if abs(delta) < 5:
        return image

    direction = "right" if delta > 0 else "left"
    prompt = f"Rotate the camera {direction} by {abs(delta)} degrees"

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = pipe(image=pil_img, prompt=prompt).images[0]

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
