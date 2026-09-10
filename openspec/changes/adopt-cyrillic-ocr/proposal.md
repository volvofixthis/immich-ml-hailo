## Why

The current OCR recognition HEF is a generic PP-OCRv5 model and does not provide reliable Russian/Cyrillic recognition. PaddleOCR provides a dedicated `cyrillic_PP-OCRv5_mobile_rec` model, and the Hailo compiler is available on the target build machine to produce a matching Hailo-10H HEF.

## What Changes

- Add a Cyrillic PP-OCRv5 recognition model and matching character dictionary.
- Export and compile the model for Hailo-10H with the Hailo Dataflow Compiler.
- Adapt the Hailo OCR recipe, calibration data, output dimensions, and CTC decoding assets.
- Keep the existing Hailo text detector and replace only the recognition stage.
- Validate Russian, English, and numeric OCR through the worker API.

## Capabilities

### New Capabilities

- `ocr/cyrillic-recognition`: Recognize Cyrillic text through a model and dictionary matched to the compiled Hailo recognition HEF.

### Modified Capabilities

## Impact

- Affects OCR model assets, conversion tooling, OCR configuration, and CTC decoding.
- Requires Hailo Dataflow Compiler tooling on the build machine.
- Requires replacing the current recognition HEF and dictionary as a matched pair; the text detection HEF remains unchanged.
