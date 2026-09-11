#!/usr/bin/env python3
"""Run the original Paddle Cyrillic recognizer on one image crop."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from paddle import inference
from PIL import Image


def prepare_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    resized_width = min(int(48 * width / height) + 1, 320)
    image = image.resize((resized_width, 48), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (320, 48), (128, 128, 128))
    canvas.paste(image, (0, 0))
    # PaddleOCR recognition uses BGR and normalization to [-1, 1].
    array = np.asarray(canvas)[:, :, ::-1].astype(np.float32)
    array = ((array / 255.0) - 0.5) / 0.5
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir
    metadata = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
    characters = metadata["PostProcess"]["character_dict"]
    if len(characters) + 2 != 852:
        raise ValueError(f"Unexpected dictionary size: {len(characters)}")

    config = inference.Config(
        str(model_dir / "inference.json"),
        str(model_dir / "inference.pdiparams"),
    )
    config.disable_gpu()
    config.switch_ir_optim(False)
    predictor = inference.create_predictor(config)
    input_handle = predictor.get_input_handle("x")
    input_data = prepare_image(args.image)
    input_handle.reshape(input_data.shape)
    input_handle.copy_from_cpu(input_data)
    predictor.run()
    logits = predictor.get_output_handle("fetch_name_0").copy_to_cpu()[0]

    indices = np.argmax(logits, axis=1)
    probabilities = np.max(logits, axis=1)
    output = []
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
            output.append(" ")
            scores.append(float(probability))
            continue
        char_index = index - 1
        if 0 <= char_index < len(characters):
            output.append(characters[char_index])
            scores.append(float(probability))

    confidence = float(np.mean(scores)) if scores else 0.0
    print(f"text={''.join(output)!r}")
    print(f"confidence={confidence:.4f}")
    print(f"logits_shape={logits.shape}")


if __name__ == "__main__":
    main()
