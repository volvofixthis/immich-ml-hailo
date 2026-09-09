#!/usr/bin/env python3
"""Download SigLIP2 assets and extract CPU-side text embedding weights."""

import argparse
import json
import os
import urllib.request

import numpy as np
from safetensors import safe_open


BASE = "https://huggingface.co/google/siglip2-base-patch32-256/resolve/main"
IMMICH_BASE = (
    "https://huggingface.co/immich-app/ViT-B-32-SigLIP2-256__webli/resolve/main/textual"
)


def download(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    tokenizer_path = os.path.join(args.models_dir, "siglip2_tokenizer.json")
    weights_path = os.path.join(args.models_dir, "siglip2_text_weights.npz")
    source_path = os.path.join(args.models_dir, "siglip2-base-patch32-256.safetensors")
    download(f"{IMMICH_BASE}/tokenizer.json", tokenizer_path)
    download(f"{BASE}/model.safetensors", source_path)

    with safe_open(source_path, framework="numpy") as source:
        keys = list(source.keys())
        token_key = next(
            k
            for k in keys
            if k.endswith("text_model.embeddings.token_embedding.weight")
        )
        position_key = next(
            k
            for k in keys
            if k.endswith("text_model.embeddings.position_embedding.weight")
        )
        token_embedding = np.asarray(source.get_tensor(token_key), dtype=np.float32)
        positional_embedding = np.asarray(
            source.get_tensor(position_key), dtype=np.float32
        )

    if token_embedding.shape != (256000, 768):
        raise ValueError(f"Unexpected token embedding shape: {token_embedding.shape}")
    if positional_embedding.shape != (64, 768):
        raise ValueError(
            f"Unexpected positional embedding shape: {positional_embedding.shape}"
        )
    np.savez(
        weights_path,
        token_embedding=token_embedding,
        positional_embedding=positional_embedding,
    )
    print(
        json.dumps(
            {
                "weights": weights_path,
                "tokenizer": tokenizer_path,
                "token_embedding": token_embedding.shape,
                "positional_embedding": positional_embedding.shape,
            }
        )
    )


if __name__ == "__main__":
    main()
