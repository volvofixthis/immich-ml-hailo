# Russian OCR Model Adoption

This document describes how to replace the generic PP-OCRv5 recognition model
with PaddleOCR's Cyrillic model while keeping the existing Hailo text detector.

The recognition model is:

```text
cyrillic_PP-OCRv5_mobile_rec
```

It uses a `48x320` recognition crop and produces `40x852` CTC logits. The
matching dictionary contains 850 characters. PaddleOCR adds one CTC blank
class at index 0 and one trailing space class, so the dictionary and model
must always be installed together:

```text
850 dictionary entries + blank + space = 852 output classes
```

## Requirements

- Hailo Dataflow Compiler 5.3.x or newer with the `hailo10h` target
- PaddlePaddle and Paddle2ONNX for model export
- ONNX and ONNX optimizer packages
- A CUDA-capable TensorFlow environment is recommended for quantization
- HailoRT and `hailortcli` on the deployment machine

The NVIDIA driver alone is not enough for GPU quantization. TensorFlow/XLA
also needs `ptxas`, `nvlink`, and `nvvm/libdevice/libdevice.10.bc` from the
CUDA toolkit.

Verify the compiler and CUDA environment:

```bash
.venv/bin/hailo --version
nvidia-smi
nvcc --version
command -v ptxas
command -v nvlink
```

If the CUDA toolkit is not installed system-wide, the Python CUDA compiler
package can be used as an alternative:

```bash
.venv/bin/pip install 'nvidia-cuda-nvcc-cu12==12.8.93'
```

## Build the DFC Docker Image

The compiler image is separate from the runtime image. The private DFC wheel
is kept outside the repository and supplied through a small Docker build
context.

From the repository root:

```bash
mkdir -p /tmp/hailo-dfc-context
cp /home/loki/Downloads/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl \
  /tmp/hailo-dfc-context/

docker build \
  -f /home/loki/projects/immich-ml-hailo/Dockerfile.hailo-dfc \
  -t hailo-dfc:5.3.0 \
  /tmp/hailo-dfc-context
```

Verify the image:

```bash
docker run --rm hailo-dfc:5.3.0
```

Expected output includes:

```text
Hailo Dataflow Compiler v5.3.0
```

Do not use the repository as the build context for this image. The Dockerfile
expects the DFC wheel at the root of the context.

The host needs NVIDIA Container Toolkit for GPU optimization:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Download the Paddle Model

Download the Paddle inference archive outside the repository:

```bash
curl -L --fail -o /tmp/cyrillic_PP-OCRv5_mobile_rec_infer.tar \
  'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv5_mobile_rec_infer.tar'
mkdir -p /tmp/cyrillic_PP-OCRv5_mobile_rec_infer
tar -xf /tmp/cyrillic_PP-OCRv5_mobile_rec_infer.tar \
  -C /tmp
```

The archive contains `inference.json`, `inference.pdiparams`, and
`inference.yml`. The dictionary is embedded in `inference.yml` under
`PostProcess.character_dict`.

## Export and Prepare ONNX

Export the Paddle graph:

```bash
.venv/bin/paddle2onnx \
  -m /tmp/cyrillic_PP-OCRv5_mobile_rec_infer \
  -mf inference.json \
  -pf inference.pdiparams \
  -s /tmp/cyrillic_PP-OCRv5_mobile_rec.onnx \
  -ov 17
```

The raw Paddle2ONNX graph is not suitable for Hailo parsing. It contains
dynamic dimensions and inference-only `Identity` nodes around the attention
blocks. Hailo's ONNX translator can mistake those Identity nodes for constant
MatMul weights.

Prepare the graph with the repository tool:

```bash
.venv/bin/python scripts/prepare_cyrillic_ocr.py \
  --input /tmp/cyrillic_PP-OCRv5_mobile_rec.onnx \
  --output /tmp/cyrillic_prepared.onnx \
  --source-yaml /tmp/cyrillic_PP-OCRv5_mobile_rec_infer/inference.yml \
  --dictionary /tmp/cyrillic_dict.txt
```

The preparation step verifies:

- Static input shape `1x3x48x320`
- Output shape `1x40x852`
- No remaining `Identity` nodes
- Dictionary size of 850 entries

## Parse for Hailo-10H

Use the Model Zoo-style recipe and parse the prepared graph:

```bash
.venv/bin/hailo parser onnx /tmp/cyrillic_prepared.onnx \
  --net-name cyrillic_ppocrv5_mobile_rec \
  --har-path /tmp/cyrillic_prepared.har \
  --hw-arch hailo10h \
  --input-format NCHW \
  --disable-onnx-simplifier \
  --parsing-report-path /tmp/cyrillic_prepared_report.json \
  -y
```

Successful parsing confirms graph compatibility only. It does not produce a
deployable HEF yet.

## Build Calibration Data

Calibration images must be preprocessed recognition crops with shape
`N x 48 x 320 x 3`. Include samples containing:

- Russian Cyrillic text
- Mixed Cyrillic and Latin text
- Digits and punctuation
- Different fonts, sizes, contrast, and crop widths
- Empty or mostly blank crops

The temporary development calibration set used during bring-up was:

```text
/tmp/cyrillic_calibration.npy
```

It contains 64 synthetic samples and is only suitable for compiler bring-up,
not final accuracy validation.

It can be regenerated with:

```bash
.venv/bin/python scripts/create_cyrillic_calibration.py \
  --output /tmp/cyrillic_calibration.npy \
  --count 64
```

The generator cycles through Russian, mixed Cyrillic/Latin, numeric, and
punctuation strings. It varies foreground/background contrast and text
position, then writes uint8 RGB crops in the required `N x 48 x 320 x 3`
layout. The Hailo recipe applies the `[127.5, 127.5, 127.5]` normalization
during optimization.

For a production HEF, replace this with at least several hundred real crops
from the target camera/library. Keep the same resize, right-padding, color
order, and dtype as the worker's recognition input.

One public source for Russian calibration images is the MIT-licensed
[GOST Russian technical OCR dataset](https://github.com/Mkz-Prog/gost-ru-technical-ocr-dataset).
It contains 1,000 labeled synthetic line images with Russian technical terms,
part numbers, measurements, punctuation, and broad Cyrillic coverage. Clone
it outside the repository with Git LFS, then convert the images:

```bash
git clone --depth 1 \
  https://github.com/Mkz-Prog/gost-ru-technical-ocr-dataset.git \
  /tmp/gost-ru-technical-ocr-dataset
git -C /tmp/gost-ru-technical-ocr-dataset lfs install --local
git -C /tmp/gost-ru-technical-ocr-dataset lfs pull

.venv/bin/python scripts/create_calibration_from_images.py \
  --images-dir /tmp/gost-ru-technical-ocr-dataset/ocr_dataset \
  --output /tmp/gost_ru_cyrillic_calibration.npy
```

Mix this dataset with real crops from the Immich workload before the final
level-4 quantization build. Its labels can also be used for held-out CER/WER
evaluation, although labels are not consumed by calibration itself.

## Quantize

The development recipe uses optimization level 1:

```text
scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls
```

Run it with CUDA enabled:

```bash
mkdir -p /tmp/cyrillic_fast_opt

.venv/bin/hailo optimize \
  /tmp/cyrillic_prepared.har \
  --hw-arch hailo10h \
  --calib-set-path /tmp/cyrillic_calibration.npy \
  --model-script scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls \
  --output-har-path /tmp/cyrillic_fast_optimized.har \
  --work-dir /tmp/cyrillic_fast_opt
```

The same level-1 optimization can run in the DFC container. The repository
and `/tmp` are mounted so the container can use the prepared HAR, calibration
array, and compiler outputs:

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/workspace" \
  -v /tmp:/tmp \
  -w /workspace \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  hailo-dfc:5.3.0 \
  bash -lc '
    mkdir -p /tmp/cyrillic_fast_opt &&
    hailo optimize \
      /tmp/cyrillic_prepared.har \
      --hw-arch hailo10h \
      --calib-set-path /tmp/cyrillic_calibration.npy \
      --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls \
      --output-har-path /tmp/cyrillic_fast_optimized.har \
      --work-dir /tmp/cyrillic_fast_opt
  '
```

`CUDA_VISIBLE_DEVICES=''` hides all CUDA GPUs from the command and forces
TensorFlow/XLA to use the CPU. Use it only when CUDA compilation is not
configured or GPU optimization is failing. It does not disable Hailo-10H;
the Hailo target is still selected by `--hw-arch hailo10h`.

For example, force CPU optimization with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/hailo optimize \
  /tmp/cyrillic_prepared.har \
  --hw-arch hailo10h \
  --calib-set-path /tmp/cyrillic_calibration.npy \
  --model-script scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls \
  --output-har-path /tmp/cyrillic_fast_optimized.har \
  --work-dir /tmp/cyrillic_fast_opt
```

For GPU optimization, omit the prefix and ensure the CUDA toolkit provides
`ptxas`, `nvlink`, and `nvvm/libdevice/libdevice.10.bc`.

Level 1 is intended to prove quantization and compilation. It skips expensive
Adaround and quantization-aware fine-tuning.

For the production build, use:

```text
scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls
```

This is the level-4 Model Zoo-style recipe. It performs more aggressive
optimization and can take substantially longer, especially on CPU.

Adaround creates large temporary caches. Do not place the optimization work
directory on a small `/tmp` tmpfs. The Makefile uses `.work/cyrillic-ocr` on
the repository filesystem by default. Override it when needed:

```bash
make WORK_DIR=/path/with/at-least-20G-free optimize-level4
```

Use the real Russian calibration array for the level-4 build:

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/workspace" \
  -v /tmp:/tmp \
  -w /workspace \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  hailo-dfc:5.3.0 \
  bash -lc '
    mkdir -p /tmp/cyrillic_opt &&
    hailo optimize \
      /tmp/cyrillic_prepared.har \
      --hw-arch hailo10h \
      --calib-set-path /tmp/gost_ru_cyrillic_calibration.npy \
      --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls \
      --output-har-path /tmp/cyrillic_optimized.har \
      --work-dir /tmp/cyrillic_opt
  '
```

## Compile the HEF

Compile the optimized HAR for Hailo-10H:

```bash
mkdir -p /tmp/cyrillic_compile

.venv/bin/hailo compiler \
  /tmp/cyrillic_fast_optimized.har \
  --hw-arch hailo10h \
  --model-script scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls \
  --output-dir /tmp/cyrillic_compile
```

Compile the level-4 HAR in the DFC container:

```bash
docker run --rm --shm-size=8g \
  -v "$PWD:/workspace" \
  -v /tmp:/tmp \
  -w /workspace \
  hailo-dfc:5.3.0 \
  bash -lc '
    mkdir -p /tmp/cyrillic_compile &&
    hailo compiler \
      /tmp/cyrillic_optimized.har \
      --hw-arch hailo10h \
      --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls \
      --output-dir /tmp/cyrillic_compile
  '
```

Do not compile the parsed or full-precision HAR. Hailo hardware requires
quantized weights.

On a system with HailoRT installed, inspect the result:

```bash
hailortcli parse-hef /tmp/cyrillic_compile/cyrillic_ppocrv5_mobile_rec.hef
```

Confirm that the input is `48x320` and the final class dimension is `852`.

## Install the Matched Assets

Copy the HEF and dictionary to the runtime model directory using the names
expected by the worker:

```text
models/cyrillic_ppocrv5_mobile_recognition.hef
models/cyrillic_ppocrv5_mobile_rec_dict.txt
```

The dictionary is the file generated by `prepare_cyrillic_ocr.py`. Never pair
the Cyrillic HEF with `ppocrv5_dict.txt` from the generic recognition model.

Enable the backend explicitly:

```bash
OCR_RECOGNITION_BACKEND=cyrillic
```

The worker validates both the dictionary length and the HEF output class
count before enabling recognition. Missing assets or a class-count mismatch
causes startup to fail when the Cyrillic backend is selected.

The detector remains the existing:

```text
models/paddle_ocr_v5_mobile_detection.hef
```

## Validation

Before deployment, compare CPU and Hailo output for:

1. Russian text, such as `Привет мир`
2. Mixed text, such as `Hello мир 123`
3. Numbers and punctuation
4. Empty or blank crops

Check text, confidence, reading order, and character error rate. Then run the
worker API OCR test with the Hailo detector and Cyrillic recognizer together.

The current repository state has completed graph preparation, Hailo parsing,
full-precision optimization, and worker validation. A quantized HEF is still
required before enabling `OCR_RECOGNITION_BACKEND=cyrillic` in production.
