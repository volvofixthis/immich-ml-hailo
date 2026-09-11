#!/usr/bin/env python3
"""Run a RapidOCR PP-OCRv5 ONNX recognizer on one text-line crop."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def prepare(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    output_width = max(320, int(np.ceil(48 * width / height)))
    image = image.resize((output_width, 48), Image.Resampling.BICUBIC)
    array = np.asarray(image)[:, :, ::-1].astype(np.float32)
    array = ((array / 255.0) - 0.5) / 0.5
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    characters = metadata["character"].splitlines()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: prepare(args.image)})[0][0]

    indices = np.argmax(output, axis=1)
    probabilities = np.max(output, axis=1)
    text = []
    scores = []
    previous = -1
    for index, probability in zip(indices, probabilities):
        index = int(index)
        if index == previous:
            continue
        previous = index
        if index == 0:
            continue
        if index == len(characters) + 1:
            text.append(" ")
            scores.append(float(probability))
            continue
        character_index = index - 1
        if 0 <= character_index < len(characters):
            text.append(characters[character_index])
            scores.append(float(probability))

    print(f"text={''.join(text)!r}")
    print(f"confidence={float(np.mean(scores)) if scores else 0.0:.4f}")
    print(f"logits_shape={output.shape}")


if __name__ == "__main__":
    main()
