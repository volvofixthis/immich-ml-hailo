"""Centralized model configuration.

All model-specific parameters (paths, layer names, quantization params) live here.
To swap a model, change the config — not the inference code.

Alternative model configs are provided as commented examples. To use them:
1. Download the HEF from the Hailo model zoo
2. Inspect it with: python3 -m ml_target.hef_inspect /app/models/<model>.hef
3. Update the config dataclass with the correct layer names and quant params
4. Rebuild the Docker image
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
CLIP_BACKEND = os.environ.get("CLIP_BACKEND", "siglip2").lower()
OCR_RECOGNITION_BACKEND = os.environ.get("OCR_RECOGNITION_BACKEND", "generic").lower()


@dataclass
class ScrfdConfig:
    """Face detection model config.

    Default: scrfd_2.5g (76.4 mAP, 1058 FPS on Hailo-8)
    Alternative: scrfd_10g (82.1 mAP, 440 FPS) — higher accuracy, lower throughput.
      To use: download scrfd_10g.hef, inspect with hef_inspect.py, update output_layers.
    """

    hef: str = "scrfd_2.5g.hef"
    input_size: int = 640
    # (stride, cls_layer_name, box_layer_name) — must match the compiled HEF
    output_layers: List[Tuple[int, str, str, str]] = field(
        default_factory=lambda: [
            (8, "scrfd_2_5g/conv42", "scrfd_2_5g/conv43", "scrfd_2_5g/conv44"),
            (16, "scrfd_2_5g/conv49", "scrfd_2_5g/conv50", "scrfd_2_5g/conv51"),
            (32, "scrfd_2_5g/conv55", "scrfd_2_5g/conv56", "scrfd_2_5g/conv57"),
        ]
    )


@dataclass
class ArcfaceConfig:
    hef: str = "arcface_r50.hef"
    crop_size: int = 112


@dataclass
class ClipImageConfig:
    """CLIP image encoder config.

    Default: TinyCLIP ViT-39M/16 (512-dim, fast, good quality)
    Alternative: SigLIP2 B/16 (siglip2_b_16_256.hef) — newer architecture,
      better zero-shot performance. Requires different tokenizer (SentencePiece
      instead of BPE) and new text_weights.npz. Input may be 256x256.
    """

    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef"
    crop_size: int = 224
    embed_dim: int = 512


@dataclass
class ClipTextConfig:
    """CLIP text encoder config.

    Default: TinyCLIP ViT-39M/16 text encoder (BPE tokenizer, UINT16 quantized)
    Alternative: SigLIP2 B/16 text (siglip2_b_16_256_text.hef) — requires
      SentencePiece tokenizer and different weights/quant params.
    """

    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef"
    weights_npz: str = "tinyclip_text_weights.npz"
    bpe_gz: str = "bpe_simple_vocab_16e6.txt.gz"
    context_length: int = 77
    embed_dim: int = 512
    # Quantization params extracted from the specific HEF via hef_inspect.py
    qp_scale: float = 3.146522067254409e-05
    qp_zp: float = 15216.0


@dataclass
class Siglip2ImageConfig:
    hef: str = "siglip2_b_32_256_image_encoder.hef"
    input_size: int = 256
    embed_dim: int = 768


@dataclass
class Siglip2TextConfig:
    hef: str = "siglip2_b_32_256_text_encoder.hef"
    weights_npz: str = "siglip2_text_weights.npz"
    tokenizer_json: str = "siglip2_tokenizer.json"
    context_length: int = 64
    embed_dim: int = 768
    qp_scale: float | None = (
        float(os.environ["SIGLIP2_TEXT_QP_SCALE"])
        if "SIGLIP2_TEXT_QP_SCALE" in os.environ
        else None
    )
    qp_zp: float | None = (
        float(os.environ["SIGLIP2_TEXT_QP_ZP"])
        if "SIGLIP2_TEXT_QP_ZP" in os.environ
        else None
    )


@dataclass
class OcrDetectionConfig:
    """PaddleOCR v5 text detection (DBNet with PPLCNetV3 backbone).

    Input: 544x960 UINT8 RGB (normalization baked into HEF).
    Output: 544x960x1 probability map (sigmoid, each pixel = text probability).
    """

    hef: str = "paddle_ocr_v5_mobile_detection.hef"
    input_h: int = 544
    input_w: int = 960
    # DBNet post-processing thresholds
    binary_thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    min_size: int = 3
    max_candidates: int = 1000


@dataclass
class OcrRecognitionConfig:
    """PaddleOCR v5 text recognition (SVTR_LCNet with CTC head).

    Input: 48x320 UINT8 RGB (normalization baked into HEF).
    Output: 1x40x18385 CTC logits (40 time steps, 18385 classes).
    Character dictionary: ppocrv5_dict.txt (18383 chars + blank + space).
    """

    hef: str = "paddle_ocr_v5_mobile_recognition.hef"
    input_h: int = 48
    input_w: int = 320
    char_dict: str = "ppocrv5_dict.txt"
    expected_classes: int = 18385
    # PaddleOCR reserves blank and unknown at output indices 0 and 1.
    ignored_indices: Tuple[int, ...] = (0, 1)


@dataclass
class CyrillicOcrRecognitionConfig(OcrRecognitionConfig):
    """Cyrillic PP-OCRv5 assets compiled from the matching 852-class graph."""

    hef: str = "cyrillic_ppocrv5_mobile_recognition.hef"
    char_dict: str = "cyrillic_ppocrv5_mobile_rec_dict.txt"
    expected_classes: int = 852


@dataclass
class PipelineConfig:
    models_dir: str = MODELS_DIR
    scrfd: ScrfdConfig = field(default_factory=ScrfdConfig)
    arcface: ArcfaceConfig = field(default_factory=ArcfaceConfig)
    clip_image: ClipImageConfig = field(default_factory=ClipImageConfig)
    clip_text: ClipTextConfig = field(default_factory=ClipTextConfig)
    siglip2_image: Siglip2ImageConfig = field(default_factory=Siglip2ImageConfig)
    siglip2_text: Siglip2TextConfig = field(default_factory=Siglip2TextConfig)
    ocr_detection: OcrDetectionConfig = field(default_factory=OcrDetectionConfig)
    ocr_recognition: OcrRecognitionConfig = field(default_factory=OcrRecognitionConfig)
    cyrillic_ocr_recognition: CyrillicOcrRecognitionConfig = field(
        default_factory=CyrillicOcrRecognitionConfig
    )

    def hef_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)
