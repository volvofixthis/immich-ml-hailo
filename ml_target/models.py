"""Hailo model wrapper: load HEF, configure network groups, run inference.

Uses the InferVStreams API (stable across HailoRT 4.x). The newer InferModel API
(available since HailoRT 4.17+) is simpler but has a different batching model:

    infer_model = vdevice.create_infer_model(hef)
    infer_model.output().set_format_type(FormatType.FLOAT32)
    configured = infer_model.configure()
    bindings = configured.create_bindings()
    bindings.input().set_buffer(input_data)
    bindings.output().set_buffer(output_buffer)
    configured.run(bindings)  # or run_async()

Migration to InferModel would simplify activate_model() since it manages
activation internally, but requires testing with actual hardware.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional

import numpy as np
import hailo_platform as hpf

LOG = logging.getLogger("ml_target.models")


@dataclass
class HailoModel:
    """Wraps a configured Hailo network group with its vstream parameters."""

    name: str
    hef_path: str
    infer_model: Any
    configured: Any
    in_key: str
    output_shapes: Dict[str, tuple[int, ...]]
    input_format: hpf.FormatType
    output_format: hpf.FormatType
    input_qp_scale: Optional[float] = None
    input_qp_zp: Optional[float] = None


def configure_model(
    vdevice: hpf.VDevice,
    hef_path: str,
    *,
    input_format: Optional[hpf.FormatType] = None,
    output_format: hpf.FormatType = hpf.FormatType.FLOAT32,
) -> HailoModel:
    """Load a HEF and configure it on the given VDevice."""
    LOG.info("Loading HEF: %s", hef_path)
    hef = hpf.HEF(hef_path)
    try:
        LOG.info(
            "  HEF inputs: %s",
            [
                (i.name, i.shape, i.format.type, i.format.order)
                for i in hef.get_input_vstream_infos()
            ],
        )
        LOG.info(
            "  HEF outputs: %s",
            [
                (o.name, o.shape, o.format.type, o.format.order)
                for o in hef.get_output_vstream_infos()
            ],
        )
    except Exception:
        LOG.exception("  Could not inspect HEF metadata: %s", hef_path)

    if input_format is None:
        input_format = _guess_input_format(hef)

    LOG.info(
        "  Creating InferModel: %s (input=%s, output=%s)",
        hef_path,
        input_format,
        output_format,
    )
    try:
        infer_model = vdevice.create_infer_model(hef_path)
        for stream in infer_model.inputs:
            stream.set_format_type(input_format)
        for stream in infer_model.outputs:
            stream.set_format_type(output_format)
        configured = infer_model.configure()
    except Exception as exc:
        LOG.exception("  Hailo configuration failed for %s", hef_path)
        raise RuntimeError(f"Hailo configuration failed for {hef_path}: {exc}") from exc

    if len(infer_model.inputs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 input stream, got {len(infer_model.inputs)}"
        )

    in_key = infer_model.inputs[0].name
    output_shapes = {stream.name: tuple(stream.shape) for stream in infer_model.outputs}
    name = in_key.split("/")[0] if "/" in in_key else in_key
    input_qp_scale, input_qp_zp = _get_input_quantization(hef)

    LOG.info("Configured model: %s", hef_path)
    LOG.info("  in_key=%s  out_keys=%s", in_key, list(output_shapes))
    LOG.info("  in_format=%s  out_format=%s", input_format, output_format)

    return HailoModel(
        name=name,
        hef_path=hef_path,
        infer_model=infer_model,
        configured=configured,
        in_key=in_key,
        output_shapes=output_shapes,
        input_format=input_format,
        output_format=output_format,
        input_qp_scale=input_qp_scale,
        input_qp_zp=input_qp_zp,
    )


def _get_input_quantization(hef: hpf.HEF) -> tuple[Optional[float], Optional[float]]:
    """Read quantization metadata when exposed by the installed HailoRT version."""
    try:
        info = hef.get_input_vstream_infos()[0]
        quant = getattr(info, "quant_info", None)
        if quant is None:
            return None, None
        scale = getattr(quant, "qp_scale", getattr(quant, "scale", None))
        zp = getattr(quant, "qp_zp", getattr(quant, "zero_point", None))
        return (
            float(scale) if scale is not None else None,
            float(zp) if zp is not None else None,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None, None


def _guess_input_format(hef: hpf.HEF) -> hpf.FormatType:
    try:
        infos = hef.get_input_vstream_infos()
        if infos and len(infos) == 1:
            s = str(
                getattr(infos[0], "format", None)
                or getattr(infos[0], "data_type", None)
                or getattr(infos[0], "dtype", None)
            ).lower()
            if "uint16" in s:
                return hpf.FormatType.UINT16
            if "float" in s:
                return hpf.FormatType.FLOAT32
    except Exception:
        pass
    return hpf.FormatType.UINT8


def validate_input(xb: np.ndarray) -> np.ndarray:
    if xb.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D input, got {xb.shape}")
    if xb.dtype not in (np.uint8, np.float32, np.uint16):
        raise ValueError(f"Expected uint8/float32/uint16, got {xb.dtype}")
    if not xb.flags["C_CONTIGUOUS"]:
        xb = np.ascontiguousarray(xb)
    return xb


@contextmanager
def activate_model(model: HailoModel) -> Generator:
    """Activate a network group for the duration of the context.

    Yields a function: infer(xb) -> Dict[str, np.ndarray].
    All inferences within the context share a single activation cycle.
    """

    def _infer(xb: np.ndarray) -> Dict[str, np.ndarray]:
        xb = validate_input(xb)
        bindings = model.configured.create_bindings()
        bindings.input().set_buffer(xb)
        output_buffers: Dict[str, np.ndarray] = {}
        for name, shape in model.output_shapes.items():
            output = np.empty((xb.shape[0], *shape), dtype=np.float32)
            bindings.output(name).set_buffer(output)
            output_buffers[name] = output
        model.configured.run([bindings], timeout=timedelta(seconds=30))
        return output_buffers

    yield _infer


def infer_single(model: HailoModel, xb: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience: activate model, run one inference, deactivate."""
    xb = validate_input(xb)
    with activate_model(model) as infer:
        return infer(xb)


def pick_output(
    outputs: Dict[str, np.ndarray], hint: Optional[str] = None
) -> np.ndarray:
    """Select the main output tensor from an inference result dict.

    If there's only one output, return it. Otherwise use `hint` to find a matching key.
    """
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    if hint:
        for k in outputs:
            if hint in k:
                return outputs[k]
    return next(iter(outputs.values()))
