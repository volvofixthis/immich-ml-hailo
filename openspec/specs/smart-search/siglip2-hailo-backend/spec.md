# siglip2-hailo-backend Specification

## Purpose

Provide a compatible Hailo-10H Smart Search backend that generates image and text embeddings from the same SigLIP2 model space.

## Requirements

### Requirement: Generate compatible SigLIP2 embeddings

The worker SHALL support SigLIP2 B/32-256 image and text embedding requests through the Hailo-10H accelerator and SHALL return 768-dimensional normalized vectors for both modalities.

#### Scenario: Image embedding

- **WHEN** a valid image request selects the visual CLIP operation
- **THEN** the worker returns a normalized 768-dimensional embedding and the original image dimensions

#### Scenario: Text embedding

- **WHEN** a text request selects the textual CLIP operation
- **THEN** the worker returns a normalized 768-dimensional embedding without image dimensions

#### Scenario: Cross-modal compatibility

- **WHEN** an image and a related text description are embedded by the worker
- **THEN** their vectors are suitable for cosine-similarity search in the same embedding space

### Requirement: Support Hailo-10H runtime execution

The worker SHALL initialize and execute the configured Hailo models using the Hailo-10H-compatible runtime interface and SHALL report a startup failure with the affected model when configuration or inference cannot be performed.

#### Scenario: Successful startup

- **WHEN** Hailo-10H is available and all required HEFs and text assets exist
- **THEN** the worker starts and exposes its health endpoint

#### Scenario: Missing or invalid model asset

- **WHEN** a required HEF or SigLIP2 text asset is missing or invalid
- **THEN** startup or the affected request fails with an actionable asset-specific error

### Requirement: Use model-specific preprocessing

The worker SHALL apply SigLIP2-compatible 256x256 RGB image preprocessing and 64-token text preprocessing, including the quantization required by the Hailo text input.

#### Scenario: Text tokenizer and embedding preparation

- **WHEN** a text query is tokenized
- **THEN** the worker produces a 64-position SigLIP2 input using the matching tokenizer and token embedding assets

#### Scenario: Image preprocessing

- **WHEN** an image is embedded
- **THEN** the worker sends a 256x256 RGB tensor in the format expected by the SigLIP2 image HEF
