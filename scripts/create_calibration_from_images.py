#!/usr/bin/env python3
"""Convert OCR line images into a Hailo calibration array."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    resized_width = min(int(48 * width / height) + 1, 320)
    image = image.resize((resized_width, 48), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (320, 48), (128, 128, 128))
    canvas.paste(image, (0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.count is not None:
        paths = paths[: args.count]
    if not paths:
        raise ValueError(f"No image files found in {args.images_dir}")

    calibration = np.stack([prepare(path) for path in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, calibration)
    print(f"saved {args.output}: shape={calibration.shape} dtype={calibration.dtype}")


if __name__ == "__main__":
    main()
