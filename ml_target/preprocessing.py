"""Image and text preprocessing for Hailo inference pipelines."""

import logging
import math
from typing import Tuple

import numpy as np

LOG = logging.getLogger("ml_target.preprocessing")


def resize_rgb(img_rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.uint8, copy=False
        )
    except Exception:
        from PIL import Image

        return np.asarray(
            Image.fromarray(img_rgb).resize((w, h), resample=Image.BILINEAR),
            dtype=np.uint8,
        )


def letterbox_rgb(
    img_rgb: np.ndarray,
    new_w: int,
    new_h: int,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox resize to (new_h, new_w), preserving aspect ratio.

    Returns (padded_image, scale, pad_x, pad_y).
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC, got {img_rgb.shape}")

    h, w = img_rgb.shape[:2]
    scale = min(new_w / w, new_h / h)
    rw = int(round(w * scale))
    rh = int(round(h * scale))

    resized = resize_rgb(img_rgb, rw, rh)

    out = np.full((new_h, new_w, 3), pad_value, dtype=np.uint8)
    pad_x = (new_w - rw) // 2
    pad_y = (new_h - rh) // 2
    out[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized

    LOG.info(
        "letterbox: orig=(%d,%d) new=(%d,%d) scale=%.6f pad=(%d,%d)",
        w,
        h,
        new_w,
        new_h,
        scale,
        pad_x,
        pad_y,
    )
    return out, scale, pad_x, pad_y


def center_crop_square(img_rgb: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square then resize to size x size."""
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img_rgb[y0 : y0 + side, x0 : x0 + side]
    return resize_rgb(crop, size, size)


def crop_and_resize_rgb(
    img_rgb: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    out_size: int,
) -> np.ndarray:
    """Crop a bounding box (with clipping) then resize to (out_size, out_size)."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    x1i = max(0, int(math.floor(x1)))
    y1i = max(0, int(math.floor(y1)))
    x2i = min(w, int(math.ceil(x2)))
    y2i = min(h, int(math.ceil(y2)))

    if x2i <= x1i or y2i <= y1i:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    crop = img_rgb[y1i:y2i, x1i:x2i]
    return resize_rgb(crop, out_size, out_size)


def align_face_rgb(
    img_rgb: np.ndarray, detection: dict, out_size: int = 112
) -> np.ndarray:
    """Warp five SCRFD landmarks to ArcFace's canonical 112px template."""
    import cv2

    landmarks = np.asarray(detection.get("landmarks", []), dtype=np.float32)
    if landmarks.shape != (5, 2):
        return crop_and_resize_rgb(img_rgb, tuple(detection["box"]), out_size)
    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    if out_size != 112:
        template *= out_size / 112.0
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if matrix is None:
        return crop_and_resize_rgb(img_rgb, tuple(detection["box"]), out_size)
    return cv2.warpAffine(img_rgb, matrix, (out_size, out_size), borderValue=(0, 0, 0))


def prep_clip_image(img_rgb_u8: np.ndarray, crop_size: int, input_format) -> np.ndarray:
    """Preprocess an image for CLIP visual encoder."""
    import hailo_platform as hpf

    img = center_crop_square(img_rgb_u8, crop_size)

    if input_format == hpf.FormatType.UINT8:
        return np.ascontiguousarray(img[None, ...], dtype=np.uint8)

    x = img.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x[None, ...], dtype=np.float32)


def prep_siglip2_image(img_rgb_u8: np.ndarray, size: int) -> np.ndarray:
    """Prepare the raw RGB UINT8 tensor expected by the SigLIP2 image HEF."""
    try:
        import cv2  # type: ignore

        img = cv2.resize(img_rgb_u8, (size, size), interpolation=cv2.INTER_CUBIC)
    except Exception:
        from PIL import Image

        img = np.asarray(
            Image.fromarray(img_rgb_u8).resize(
                (size, size), resample=Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(img[None, ...], dtype=np.uint8)


def prep_clip_text_input(
    token_ids_77: np.ndarray,
    token_embedding: np.ndarray,
    positional_embedding: np.ndarray,
    qp_scale: float,
    qp_zp: float,
) -> np.ndarray:
    """Pack token embeddings into quantized UINT16 input for text encoder HEF."""
    if token_ids_77.shape != (77,):
        raise ValueError(f"token_ids must be (77,), got {token_ids_77.shape}")

    x = token_embedding[token_ids_77] + positional_embedding  # (77, 512) float32
    x_u16 = np.clip(np.round(x / qp_scale + qp_zp), 0, 65535).astype(np.uint16)
    return np.ascontiguousarray(x_u16[None, None, ...], dtype=np.uint16)


def prep_siglip2_text_input(
    token_ids: np.ndarray,
    token_embedding: np.ndarray,
    qp_scale: float,
    qp_zp: float,
) -> np.ndarray:
    """Build the token-embedding input to the SigLIP2 text HEF."""
    if token_ids.shape != (64,):
        raise ValueError(f"token_ids must be (64,), got {token_ids.shape}")
    if token_embedding.ndim != 2 or token_embedding.shape[1] != 768:
        raise ValueError(f"unexpected token embedding shape: {token_embedding.shape}")
    if qp_scale <= 0:
        raise ValueError(f"quantization scale must be positive, got {qp_scale}")

    # The Hailo graph starts at the model's embeddings/Add node and contains
    # the positional embedding internally. Adding it here would double it.
    x = token_embedding[token_ids]
    x_u16 = np.clip(np.round(x / qp_scale + qp_zp), 0, 65535).astype(np.uint16)
    return np.ascontiguousarray(x_u16[None, ...], dtype=np.uint16)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False).reshape(-1)
    return v / (float(np.linalg.norm(v)) + 1e-12)
