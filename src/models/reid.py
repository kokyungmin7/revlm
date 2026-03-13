"""Person Re-Identification using DINOv2 + ArcFace.

Model: DavronSherbaev/person-reid-arcface
  - Backbone: DINOv2 ViT-B/14, outputs 768-dim CLS token
  - Head: ArcFace additive angular margin classifier (15 office classes)
  - Inference output: L2-normalized 768-dim embedding vector

Usage:
    model, device = load_reid_model()
    emb = extract_embedding(model, bgr_crop, device)
    sim = compute_similarity(emb_a, emb_b)  # cosine similarity in [-1, 1]
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

_REPO_ID = "DavronSherbaev/person-reid-arcface"
_FILENAME = "arcface_dinov2.pth"
_EMBED_DIM = 768
_NUM_CLASSES = 15
_MARGIN = 0.5

_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ArcMarginProduct(nn.Module):
    """ArcFace additive angular margin product layer.

    Used only during training; kept here to match the saved checkpoint structure.

    Args:
        in_features: Embedding dimension (768).
        out_features: Number of classes (15).
        s: Feature scale factor.
        m: Angular margin in radians.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.s


class ArcFaceReID(nn.Module):
    """DINOv2 ViT-B/14 backbone with ArcFace classification head.

    For inference, use get_embedding() to extract L2-normalized 768-dim vectors.

    Args:
        num_classes: Number of person classes in the training set.
        embed_dim: Embedding dimension of the backbone output.
        margin: ArcFace angular margin.
    """

    def __init__(
        self,
        num_classes: int = _NUM_CLASSES,
        embed_dim: int = _EMBED_DIM,
        margin: float = _MARGIN,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=False,
        )
        self.arc_head = ArcMarginProduct(embed_dim, num_classes, m=margin)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embedding from an image batch.

        Args:
            x: Image tensor of shape (B, 3, 224, 224).

        Returns:
            L2-normalized embedding tensor of shape (B, 768).
        """
        features = self.backbone(x)  # (B, 768) CLS token
        return F.normalize(features, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.get_embedding(x)
        if labels is not None:
            return self.arc_head(emb, labels)
        return emb


def _load_state_dict(model: ArcFaceReID, state_dict: dict) -> None:
    """Load state dict with fallback strategies for key mismatches.

    Tries in order:
    1. Strict load (exact key match)
    2. Non-strict load (ignore unexpected/missing keys)
    3. Backbone-only load (filter keys starting with 'backbone.')
    """
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # If backbone weights are missing, try extracting backbone-prefixed keys
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        backbone_sd = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        if backbone_sd:
            model.backbone.load_state_dict(backbone_sd, strict=False)


def load_reid_model(device: str | None = None) -> tuple["ArcFaceReID", str]:
    """Load the ArcFace person ReID model from HuggingFace Hub.

    Downloads and caches the checkpoint on first call.

    Args:
        device: Torch device string. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        Tuple of (model, device_string).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = ArcFaceReID()
    _load_state_dict(model, state_dict)
    model.to(device)
    model.eval()

    return model, device


def extract_embedding(
    model: "ArcFaceReID",
    image: np.ndarray,
    device: str,
) -> np.ndarray:
    """Extract ReID embedding from a BGR person crop.

    Args:
        model: Loaded ArcFaceReID model (from load_reid_model).
        image: BGR person crop (H x W x 3 uint8 ndarray).
        device: Torch device string.

    Returns:
        L2-normalized embedding vector of shape (768,).
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = _TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.get_embedding(x)

    return embedding.cpu().numpy().squeeze()  # (768,)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized ReID embeddings.

    Since embeddings are L2-normalized, this is equivalent to the dot product.

    Args:
        emb1: First embedding of shape (768,).
        emb2: Second embedding of shape (768,).

    Returns:
        Similarity score in [-1.0, 1.0]. Higher means more similar.
    """
    return float(np.dot(emb1, emb2))
