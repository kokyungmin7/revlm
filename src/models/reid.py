"""Person Re-Identification model backends.

Supported models (--reid_model):
    arcface-dinov2   DavronSherbaev/person-reid-arcface  (DINOv2 ViT-B/14, 768-dim)
    siglip2          MarketaJu/siglip2-person-description-reid

Usage:
    model = load_reid_model("arcface-dinov2")   # or "siglip2"
    emb = model.extract_embedding(bgr_crop)     # np.ndarray (embed_dim,)
    sim = compute_similarity(emb_a, emb_b)

    # Backward-compatible call style:
    model, device = load_reid_model()
    emb = extract_embedding(model, bgr_crop, device)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import Protocol, runtime_checkable


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class ReIDBackend(Protocol):
    """Common interface for all ReID backends."""

    embed_dim: int

    def extract_embedding(self, bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embedding from a BGR person crop.

        Args:
            bgr: BGR uint8 ndarray of shape (H, W, 3).

        Returns:
            L2-normalized float32 embedding of shape (embed_dim,).
        """
        ...


# ── ArcFace + DINOv2 backend ──────────────────────────────────────────────────

_ARCFACE_REPO_ID = "DavronSherbaev/person-reid-arcface"
_ARCFACE_FILENAME = "arcface_dinov2.pth"

_ARCFACE_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class _ArcMarginProduct(nn.Module):
    """ArcFace additive angular margin layer (kept to match checkpoint keys)."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.s


class _ArcFaceReIDModule(nn.Module):
    """DINOv2 ViT-B/14 + ArcFace head (internal nn.Module for ArcFaceDinoV2)."""

    def __init__(self, num_classes: int = 15, embed_dim: int = 768, margin: float = 0.5) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False
        )
        self.arc_head = _ArcMarginProduct(embed_dim, num_classes, m=margin)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=-1)


def _load_arcface_state_dict(model: _ArcFaceReIDModule, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    missing, _ = model.load_state_dict(state_dict, strict=False)

    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        backbone_sd = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        if backbone_sd:
            model.backbone.load_state_dict(backbone_sd, strict=False)


class ArcFaceDinoV2:
    """ReID backend: DavronSherbaev/person-reid-arcface (DINOv2 ViT-B/14 + ArcFace).

    Output: L2-normalized 768-dim embedding.
    """

    embed_dim: int = 768

    def __init__(self, device: str) -> None:
        self._device = device
        model_path = hf_hub_download(repo_id=_ARCFACE_REPO_ID, filename=_ARCFACE_FILENAME)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self._model = _ArcFaceReIDModule()
        _load_arcface_state_dict(self._model, state_dict)
        self._model.to(device).eval()

    def extract_embedding(self, bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = _ARCFACE_TRANSFORM(Image.fromarray(img_rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.get_embedding(x)
        return emb.cpu().numpy().squeeze()  # (768,)


# ── SigLIP2 backend ───────────────────────────────────────────────────────────

class SigLIP2ReID:
    """ReID backend: MarketaJu/siglip2-person-description-reid (SigLIP2 vision encoder).

    Uses only the vision encoder (image-to-image retrieval).
    embed_dim is auto-detected from model config.
    """

    MODEL_ID = "MarketaJu/siglip2-person-description-reid"
    _FALLBACK_PROCESSOR = "google/siglip-base-patch16-224"

    def __init__(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._device = device
        self._model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        ).to(device).eval()
        self.embed_dim: int = self._model.config.vision_config.hidden_size

        # Processor: try model-attached first, then fallback to base SigLIP processor
        if hasattr(self._model, "processor") and self._model.processor is not None:
            self._processor = self._model.processor
        elif hasattr(self._model, "image_processor") and self._model.image_processor is not None:
            self._processor = self._model.image_processor
        else:
            self._processor = AutoImageProcessor.from_pretrained(self._FALLBACK_PROCESSOR)

    def extract_embedding(self, bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_embeds = self._model.get_image_features(**inputs)
            # get_image_features may return a tensor or an object with named attributes
            if hasattr(image_embeds, "pooler_output") and image_embeds.pooler_output is not None:
                image_embeds = image_embeds.pooler_output
            elif hasattr(image_embeds, "last_hidden_state"):
                image_embeds = image_embeds.last_hidden_state
            elif hasattr(image_embeds, "image_embeds"):
                image_embeds = image_embeds.image_embeds

        embedding = image_embeds.cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ── Registry & factory ────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "arcface-dinov2": ArcFaceDinoV2,
    "siglip2": SigLIP2ReID,
}


def load_reid_model(
    name: str = "arcface-dinov2",
    device: str | None = None,
) -> ReIDBackend:
    """Load a ReID backend by name.

    Args:
        name: Model key. One of: arcface-dinov2, siglip2.
        device: Torch device. Auto-detected if None.

    Returns:
        ReIDBackend instance with .embed_dim and .extract_embedding().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in _REGISTRY:
        raise ValueError(f"Unknown ReID model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](device)


# ── Backward-compatible helpers ───────────────────────────────────────────────

def extract_embedding(
    model: ReIDBackend,
    image: np.ndarray,
    device: str,
) -> np.ndarray:
    """Backward-compatible wrapper around model.extract_embedding().

    Args:
        model: Any ReIDBackend (from load_reid_model).
        image: BGR uint8 ndarray.
        device: Ignored (kept for API compatibility).

    Returns:
        L2-normalized embedding array of shape (embed_dim,).
    """
    return model.extract_embedding(image)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized ReID embeddings.

    Args:
        emb1: First embedding.
        emb2: Second embedding (must have same shape as emb1).

    Returns:
        Similarity score in [-1.0, 1.0].
    """
    return float(np.dot(emb1, emb2))
