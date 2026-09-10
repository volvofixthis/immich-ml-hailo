## 1. Model preparation

- [ ] 1.1 Install Hailo Dataflow Compiler 5.3.x and verify the compiler targets `hailo10h`.
- [ ] 1.2 Download `cyrillic_PP-OCRv5_mobile_rec` and its matching dictionary; verify the reference model recognizes Russian samples.
- [ ] 1.3 Export the Cyrillic model to ONNX and verify input, output, and CTC dimensions against the Paddle reference.

## 2. Hailo compilation

- [ ] 2.1 Adapt the PP-OCRv5 Hailo recipe for the Cyrillic ONNX graph and output vocabulary; verify the parser accepts the graph.
- [ ] 2.2 Build a representative Cyrillic calibration set and quantize for Hailo-10H; verify optimization completes without unsupported operators.
- [ ] 2.3 Compile the recognition HEF and inspect it with `hailortcli parse-hef`; verify Hailo-10H compatibility and expected tensor shapes.

## 3. Worker integration

- [ ] 3.1 Install the compiled HEF and matching dictionary as a versioned asset pair; verify dictionary length matches the HEF CTC class count.
- [ ] 3.2 Update OCR configuration and asset validation; verify the existing Hailo detector remains unchanged.
- [ ] 3.3 Compare Hailo recognition with CPU reference output on Russian, mixed Latin/Cyrillic, numeric, and empty-text samples.

## 4. Deployment and validation

- [ ] 4.1 Deploy the recognition HEF on the Raspberry Pi and verify the worker starts and `/ping` responds.
- [ ] 4.2 Run direct OCR API tests and verify Russian text, confidence scores, and reading order.
- [ ] 4.3 Run Immich OCR jobs and verify recognized text is searchable in the library.
