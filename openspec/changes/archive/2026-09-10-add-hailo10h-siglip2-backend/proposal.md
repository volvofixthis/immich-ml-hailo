## Why

The worker originally targeted Hailo-8-era runtime APIs and TinyCLIP assets, so it could not use the matching SigLIP2 image/text HEFs on Hailo-10H. This change records the completed migration needed to provide compatible 768-dimensional Smart Search embeddings.

## What Changes

- Add a selectable, default SigLIP2 B/32-256 CLIP backend for Hailo-10H.
- Use HailoRT 5.3.0 and the Hailo-10H-compatible `InferModel` API.
- Add SigLIP2 tokenizer and CPU-side token embedding preparation.
- Add 256x256 image preprocessing and 64-token text preprocessing.
- Mount model assets at runtime and provide Docker Compose startup.
- Add diagnostics and validation documentation.

## Capabilities

### New Capabilities

- `smart-search/siglip2-hailo-backend`: Generate compatible SigLIP2 image and text embeddings through Hailo-10H.

### Modified Capabilities

## Impact

- Affects `ml_target` inference, preprocessing, tokenization, configuration, and model loading.
- Updates Docker/HailoRT packaging and runtime model asset handling.
- Changes Smart Search embeddings from TinyCLIP 512-dimensional vectors to SigLIP2 768-dimensional vectors; existing embeddings must be regenerated.
