#!/usr/bin/env python3
"""Prepare PaddleOCR Cyrillic ONNX for the Hailo Model Zoo recipe.

Paddle2ONNX emits dynamic dimensions and Identity nodes around inference-only
dropout paths. Hailo's ONNX translator treats Identity as a constant while
looking for MatMul weights, so those nodes must be eliminated before parsing.
"""

import argparse
from pathlib import Path

import onnx
import onnxoptimizer
import yaml


OPTIMIZER_PASSES = [
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_nop_transpose",
    "eliminate_nop_pad",
    "eliminate_deadend",
    "fuse_consecutive_transposes",
    "fuse_consecutive_squeezes",
]


def set_static_shape(value_info: onnx.ValueInfoProto, shape: list[int]) -> None:
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != len(shape):
        raise ValueError(f"Unexpected rank for {value_info.name}: {len(dims)}")
    for dim, size in zip(dims, shape):
        dim.ClearField("dim_param")
        dim.dim_value = size


def extract_dictionary(source_yaml: Path, output_path: Path) -> int:
    metadata = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    chars = metadata["PostProcess"]["character_dict"]
    if not isinstance(chars, list) or not chars:
        raise ValueError("PostProcess.character_dict is empty or invalid")
    output_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    return len(chars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-yaml", type=Path)
    parser.add_argument("--dictionary", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    model = onnxoptimizer.optimize(model, OPTIMIZER_PASSES)
    set_static_shape(model.graph.input[0], [1, 3, 48, 320])
    set_static_shape(model.graph.output[0], [1, 40, 852])

    if any(node.op_type == "Identity" for node in model.graph.node):
        raise RuntimeError("Identity nodes remain after ONNX optimization")
    output_classes = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    if output_classes != 852:
        raise ValueError(f"Expected 852 output classes, got {output_classes}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))

    dictionary_length = None
    if args.source_yaml or args.dictionary:
        if not args.source_yaml or not args.dictionary:
            parser.error("--source-yaml and --dictionary must be supplied together")
        dictionary_length = extract_dictionary(args.source_yaml, args.dictionary)
        if dictionary_length + 2 != output_classes:
            raise ValueError(
                f"Dictionary has {dictionary_length} entries but the model has "
                f"{output_classes} classes (expected dictionary + 2 CTC tokens)"
            )

    print(
        f"prepared {args.output}: input=1x3x48x320 output=1x40x{output_classes} "
        f"nodes={len(model.graph.node)} dictionary={dictionary_length}"
    )


if __name__ == "__main__":
    main()
