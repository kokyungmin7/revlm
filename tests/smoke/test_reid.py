from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_reid_embedding_shape(sample_image_bgr: np.ndarray) -> None:
    from src.models.reid import extract_embedding, load_reid_model

    model, device = load_reid_model()
    emb = extract_embedding(model, sample_image_bgr, device)

    assert emb.ndim == 1 and emb.shape[0] == 768, (
        f"Expected embedding shape (768,), got {emb.shape}"
    )

    norm = float(np.linalg.norm(emb))
    print(f"\n[smoke_reid] embedding shape: {emb.shape}")
    print(f"  L2 norm: {norm:.4f} (should be ~1.0 — L2 normalized)")
    assert abs(norm - 1.0) < 1e-4, f"Embedding is not L2-normalized, norm={norm:.4f}"


def test_reid_same_image_similarity(sample_image_bgr: np.ndarray) -> None:
    from src.models.reid import compute_similarity, extract_embedding, load_reid_model

    model, device = load_reid_model()
    emb1 = extract_embedding(model, sample_image_bgr, device)
    emb2 = extract_embedding(model, sample_image_bgr, device)

    sim = compute_similarity(emb1, emb2)
    print(f"\n[smoke_reid] same-image cosine similarity: {sim:.6f} (expected ~1.0)")
    assert sim > 0.999, f"Identical inputs should yield similarity ~1.0, got {sim:.6f}"


def test_reid_crop_similarity_and_save(sample_image_bgr: np.ndarray) -> None:
    from src.models.reid import compute_similarity, extract_embedding, load_reid_model

    Path("experiments").mkdir(exist_ok=True)
    model, device = load_reid_model()

    h, w = sample_image_bgr.shape[:2]
    crop_top = sample_image_bgr[: h // 2, :, :]
    crop_bot = sample_image_bgr[h // 2 :, :, :]

    emb_full = extract_embedding(model, sample_image_bgr, device)
    emb_top = extract_embedding(model, crop_top, device)
    emb_bot = extract_embedding(model, crop_bot, device)

    sim_top = compute_similarity(emb_full, emb_top)
    sim_bot = compute_similarity(emb_full, emb_bot)
    sim_cross = compute_similarity(emb_top, emb_bot)

    print(f"\n[smoke_reid] full vs top-half:    {sim_top:.4f}")
    print(f"  full vs bottom-half: {sim_bot:.4f}")
    print(f"  top  vs bottom-half: {sim_cross:.4f}")

    # Annotate and save side-by-side comparison
    def _annotate(img: np.ndarray, label: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        return out

    full_ann = _annotate(sample_image_bgr, "full")
    top_ann = _annotate(crop_top, f"top  sim={sim_top:.3f}")
    bot_ann = _annotate(crop_bot, f"bot  sim={sim_bot:.3f}")

    # Pad to same height before hstack
    target_h = max(full_ann.shape[0], top_ann.shape[0], bot_ann.shape[0])

    def _pad_h(img: np.ndarray, h: int) -> np.ndarray:
        diff = h - img.shape[0]
        if diff <= 0:
            return img
        return np.vstack([img, np.zeros((diff, img.shape[1], 3), dtype=img.dtype)])

    canvas = np.hstack([
        _pad_h(full_ann, target_h),
        _pad_h(top_ann, target_h),
        _pad_h(bot_ann, target_h),
    ])
    out_path = "experiments/smoke_reid_similarity.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"  → saved: {out_path}")
