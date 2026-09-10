## Context

The current OCR detector and recognition integration targets the generic Hailo PP-OCRv5 mobile recognition HEF with an 18,385-character output vocabulary. The target build machine will provide Hailo Dataflow Compiler tooling for Hailo-10H compilation.

## Goals / Non-Goals

**Goals:**

- Compile and run `cyrillic_PP-OCRv5_mobile_rec` on Hailo-10H.
- Reuse the current 48x320 recognition preprocessing and CTC decoding contract where compatible.
- Keep text detection on the existing Hailo HEF.
- Validate Russian, mixed-language, numeric, and negative OCR cases.

**Non-Goals:**

- Replace the text detector.
- Support every PaddleOCR language model in one HEF.
- Change the Immich OCR response format.

## Decisions

- **Use the Hailo PP-OCRv5 mobile recognition recipe as the starting point.** It already establishes the supported input shape, normalization, parser settings, calibration structure, and Hailo-10H target.
- **Compile on the dedicated Hailo compiler machine.** The Raspberry Pi only needs the resulting HEF and matching dictionary; compilation and model export are not runtime responsibilities.
- **Use a Cyrillic calibration set.** Calibration images must represent Russian text, fonts, sizes, contrast, and mixed Latin/digit content so quantization does not favor the generic dataset.
- **Keep the dictionary beside the HEF.** The application should treat the pair as one versioned asset and validate output class count against dictionary length.
- **Keep a CPU recognition fallback available during development.** It provides a reference result for comparing the quantized Hailo model before enabling it in production.

## Risks / Trade-offs

- [Paddle-to-ONNX export differences] The exported graph may differ from the generic recipe → compare input/output shapes and validate against Paddle reference inference.
- [Unsupported operators] Hailo parsing may reject an export → simplify or adapt the graph while preserving the CTC head.
- [Quantization loss] Small Cyrillic glyphs may be damaged → use representative Cyrillic calibration data and compare character error rate.
- [Vocabulary mismatch] A dictionary-only replacement can silently corrupt output → package and validate the HEF/dictionary pair together.

## Migration Plan

1. Install Hailo Dataflow Compiler 5.3.x and the matching Model Zoo recipe on the compiler machine.
2. Export and validate the Cyrillic PaddleOCR model as ONNX.
3. Compile, quantize, and generate a Hailo-10H HEF using Cyrillic calibration data.
4. Validate the HEF on the Raspberry Pi with the existing OCR API and compare results with the CPU reference.
5. Replace the generic recognition HEF and dictionary, rebuild/restart the worker, and run OCR jobs again.

Rollback consists of restoring the current generic recognition HEF and dictionary pair.
