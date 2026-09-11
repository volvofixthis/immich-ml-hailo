SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT)/.venv/bin/python

DFC_IMAGE ?= hailo-dfc:5.3.0
DFC_WHL ?= $(HOME)/Downloads/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
DFC_CONTEXT ?= /tmp/hailo-dfc-context
WORK_DIR ?= $(ROOT)/.work/cyrillic-ocr

PADDLE_URL ?= https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv5_mobile_rec_infer.tar
PADDLE_TAR := $(WORK_DIR)/cyrillic_PP-OCRv5_mobile_rec_infer.tar
PADDLE_DIR := $(WORK_DIR)/cyrillic_PP-OCRv5_mobile_rec_infer
RAW_ONNX := $(WORK_DIR)/cyrillic_PP-OCRv5_mobile_rec.onnx
PREPARED_ONNX := $(WORK_DIR)/cyrillic_prepared.onnx
PREPARED_HAR := $(WORK_DIR)/cyrillic_prepared.har
DICT := $(WORK_DIR)/cyrillic_dict.txt

DATASET_DIR := $(WORK_DIR)/gost-ru-technical-ocr-dataset
CALIBRATION := $(WORK_DIR)/gost_ru_cyrillic_calibration.npy

FAST_SCRIPT := $(ROOT)/scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls
FINAL_SCRIPT := $(ROOT)/scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls
FAST_HAR := $(WORK_DIR)/cyrillic_fast_optimized.har
FINAL_HAR := $(WORK_DIR)/cyrillic_optimized.har
COMPILE_DIR ?= $(WORK_DIR)/compile

DOCKER_GPU ?= --gpus all
HOST_USER ?= $(shell id -un)
DOCKER_COMMON = --rm --shm-size=8g $(DOCKER_GPU) \
	-v "$(ROOT):/workspace" \
	-v "$(WORK_DIR):/work" \
	-w /workspace \
	-e USER=$(HOST_USER) \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility

.PHONY: help image paddle-model export prepare dataset calibration assets parse \
	optimize-level1 optimize-level4 compile-level1 compile-level4 clean-work

help:
	@printf '%s\n' \
		'make image             Build the DFC compiler image' \
		'make assets            Download and prepare model/calibration assets' \
		'make parse             Parse the prepared ONNX for Hailo-10H' \
		'make optimize-level1  Run fast development quantization' \
		'make optimize-level4  Run production quantization' \
		'make compile-level1   Compile the level-1 HAR to HEF' \
		'make compile-level4   Compile the level-4 HAR to HEF' \
		'make clean-work       Remove generated files under WORK_DIR'

image:
	@test -f "$(DFC_WHL)" || { echo "Missing DFC wheel: $(DFC_WHL)"; exit 1; }
	mkdir -p "$(WORK_DIR)"
	mkdir -p "$(DFC_CONTEXT)"
	cp "$(DFC_WHL)" "$(DFC_CONTEXT)/"

paddle-model:
	mkdir -p "$(PADDLE_DIR)"
	@if [[ ! -s "$(PADDLE_TAR)" ]]; then \
		curl -L --fail --retry 2 -o "$(PADDLE_TAR)" "$(PADDLE_URL)"; \
	fi
	@if [[ ! -f "$(PADDLE_DIR)/inference.json" ]]; then \
		tar -xf "$(PADDLE_TAR)" --strip-components=1 -C "$(PADDLE_DIR)"; \
	fi

export: image paddle-model
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'paddle2onnx -m /work/cyrillic_PP-OCRv5_mobile_rec_infer -mf inference.json -pf inference.pdiparams -s /work/cyrillic_PP-OCRv5_mobile_rec.onnx -ov 17'

prepare: image export
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'python /workspace/scripts/prepare_cyrillic_ocr.py --input /work/cyrillic_PP-OCRv5_mobile_rec.onnx --output /work/cyrillic_prepared.onnx --source-yaml /work/cyrillic_PP-OCRv5_mobile_rec_infer/inference.yml --dictionary /work/cyrillic_dict.txt'

dataset:
	@if [[ ! -d "$(DATASET_DIR)/.git" ]]; then \
		git clone --depth 1 https://github.com/Mkz-Prog/gost-ru-technical-ocr-dataset.git "$(DATASET_DIR)"; \
	fi
	git -C "$(DATASET_DIR)" lfs install --local
	git -C "$(DATASET_DIR)" lfs pull

calibration: image dataset
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'python /workspace/scripts/create_calibration_from_images.py --images-dir /work/gost-ru-technical-ocr-dataset/ocr_dataset --output /work/gost_ru_cyrillic_calibration.npy'

assets: image prepare calibration

parse: image prepare
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'hailo parser onnx /work/cyrillic_prepared.onnx --net-name cyrillic_ppocrv5_mobile_rec --har-path /work/cyrillic_prepared.har --hw-arch hailo10h --input-format NCHW --disable-onnx-simplifier --parsing-report-path /work/cyrillic_prepared_report.json -y'

optimize-level1: image parse calibration
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'mkdir -p /work/fast-opt && hailo optimize /work/cyrillic_prepared.har --hw-arch hailo10h --calib-set-path /work/gost_ru_cyrillic_calibration.npy --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls --output-har-path /work/cyrillic_fast_optimized.har --work-dir /work/fast-opt'

optimize-level4: image parse calibration
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'mkdir -p /work/final-opt && hailo optimize /work/cyrillic_prepared.har --hw-arch hailo10h --calib-set-path /work/gost_ru_cyrillic_calibration.npy --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls --output-har-path /work/cyrillic_optimized.har --work-dir /work/final-opt'

compile-level1: image
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'mkdir -p /work/compile-level1 && hailo compiler /work/cyrillic_fast_optimized.har --hw-arch hailo10h --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition_fast.alls --output-dir /work/compile-level1'

compile-level4: image 
	docker run $(DOCKER_COMMON) "$(DFC_IMAGE)" bash -lc \
		'mkdir -p /work/compile-level4 && hailo compiler /work/cyrillic_optimized.har --hw-arch hailo10h --model-script /workspace/scripts/cyrillic_paddle_ocr_v5_mobile_recognition.alls --output-dir /work/compile-level4'

clean-work:
	rm -rf "$(WORK_DIR)"
