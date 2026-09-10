## Context

The worker originally used TinyCLIP and the legacy HailoRT stream API. Hailo-10H HEFs require HailoRT 5.3.0 and the `InferModel` API. SigLIP2 also requires model-specific tokenization, CPU-side token embedding preparation, 256x256 images, and 64-token text inputs.

## Goals / Non-Goals

**Goals:**

- Run the matching SigLIP2 B/32-256 image and text HEFs on Hailo-10H.
- Preserve the worker's Immich-compatible `/predict` response contract.
- Keep model assets outside the image and mount them read-only at runtime.
- Make startup and runtime failures diagnosable.

**Non-Goals:**

- Replace the existing face or OCR models.
- Provide a Russian OCR HEF.
- Preserve compatibility between old TinyCLIP and new SigLIP2 embeddings.

## Decisions

- **Use `InferModel` instead of legacy `VDevice.configure`.** Hailo-10H with HailoRT 5.3.0 successfully configures the HEFs through `create_infer_model`, while the legacy path returns `HAILO_NOT_IMPLEMENTED`.
- **Keep a shared VDevice and configure each model once.** This allows face, OCR, image, and text requests to reuse configured network groups without reloading HEFs per request.
- **Use runtime-mounted assets.** HEFs and large CPU-side text assets are mounted from `models/` rather than copied into the application image, allowing model replacement without rebuilding the image.
- **Keep token embedding preparation on CPU.** The text HEF accepts a quantized embedding tensor rather than token IDs; the tokenizer and embedding table therefore remain host-side.
- **Run ArcFace one face at a time.** The supplied ArcFace HEF has batch size one. SCRFD landmarks are used to align each face to the ArcFace canonical template before recognition.

## Risks / Trade-offs

- [Large text embedding table] The CPU-side SigLIP2 token embedding table consumes substantial RAM → keep it as a runtime asset and consider memory-mapped storage later.
- [Embedding migration] Existing TinyCLIP vectors are incompatible → regenerate all Immich Smart Search embeddings after deployment.
- [Model asset mismatch] HEFs and tokenizer/weights can silently produce incompatible vectors → validate direct image/text cosine similarity before reindexing.
- [OCR language coverage] The published OCR recognition HEF is not a dedicated Cyrillic model → use a matching Cyrillic model/HEF or CPU recognition for Russian text.

## Migration Plan

1. Install HailoRT 5.3.0 and verify Hailo-10H visibility with `hailortcli scan`.
2. Place compatible HEFs and SigLIP2 text assets in `models/`.
3. Build the base and application images and start with Docker Compose.
4. Validate `/ping`, worker startup, embedding dimensions, and cross-modal similarity.
5. Regenerate Immich Smart Search embeddings with **Smart Search → All**.

Rollback consists of stopping the new worker and restoring the previous worker/model configuration; existing vectors should not be mixed across model families.
