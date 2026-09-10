## 1. Runtime and model support

- [x] 1.1 Update Docker and setup tooling for HailoRT 5.3.0 and verify the base image builds.
- [x] 1.2 Add Hailo-10H HEF discovery, metadata logging, and runtime-mounted model assets; verify `hailortcli scan` and `parse-hef`.
- [x] 1.3 Replace the legacy Hailo configuration path with `InferModel`; verify all configured HEFs start on Hailo-10H.

## 2. SigLIP2 backend

- [x] 2.1 Add SigLIP2 B/32-256 configuration and backend selection; verify image and text outputs are 768-dimensional.
- [x] 2.2 Add matching tokenizer and CPU-side text embedding preparation; verify 64-token input quantization.
- [x] 2.3 Add SigLIP2 image preprocessing and cross-modal validation; verify related image/text prompts produce meaningful similarity differences.

## 3. Existing model compatibility

- [x] 3.1 Update multi-output inference handling for SCRFD and OCR; verify the API test suite passes for face, OCR, and CLIP requests.
- [x] 3.2 Handle ArcFace batch size one and landmark-based face alignment; verify face inference no longer sends oversized input buffers.

## 4. Deployment and migration

- [x] 4.1 Add Docker Compose startup with PCIe access and read-only model mounts; verify `/ping` and container health.
- [x] 4.2 Document HailoRT API diagnostics, validation, and OCR language limitations; verify `docs/testing.md` covers the working commands.
- [x] 4.3 Regenerate Immich Smart Search embeddings after deployment; verify Smart Search uses the new 768-dimensional SigLIP2 vectors.
