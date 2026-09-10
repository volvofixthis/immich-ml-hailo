#!/usr/bin/env python3
"""Create a small synthetic Cyrillic OCR calibration set.

This is for compiler bring-up only. Production calibration should use real
recognition crops collected from the target workload.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TEXTS = (
    "Привет мир",
    "Москва 2026",
    "Санкт-Петербург",
    "Русский текст",
    "Тестовый номер 123",
    "Hello мир 42",
    "Цена: 123 руб",
    "АБВГД ЕЖЗИЙ",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/cyrillic_calibration.npy"))
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Regular.ttf", 28)
    images = []
    for index in range(args.count):
        background = int(rng.integers(190, 256))
        foreground = int(rng.integers(0, 70))
        image = Image.new("RGB", (320, 48), (background,) * 3)
        draw = ImageDraw.Draw(image)
        draw.text(
            (int(rng.integers(0, 12)), int(rng.integers(0, 8))),
            TEXTS[index % len(TEXTS)],
            font=font,
            fill=(foreground,) * 3,
        )
        images.append(np.asarray(image, dtype=np.uint8))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(images))
    print(f"saved {args.output}: shape={np.stack(images).shape} dtype=uint8")


if __name__ == "__main__":
    main()
