"""Core ML semantic encoder with explicit, ANE-compatible convolution caches.

The published legacy package reads MLState buffers but never writes them. Export
those buffers as tensor inputs and the tail of each causal concat as outputs.
This also avoids the state operations that fail at inference on the M5 ANE.
Core ML remains an optional dependency; conversion never loads execution devices.
"""

from __future__ import annotations

import hashlib
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def explicit_cache_spec(spec: Any) -> Any:
    """Convert the legacy read-only state graph without changing its weights."""
    from coremltools.proto import MIL_pb2 as mil

    spec = deepcopy(spec)
    function = spec.mlProgram.functions["main"]
    states = {state.name: deepcopy(state) for state in spec.description.state}
    if not states or len(function.block_specializations) != 1:
        raise ValueError("Expected the legacy single-block stateful semantic encoder")
    for state in states.values():
        feature = spec.description.input.add(name=state.name)
        feature.type.multiArrayType.CopyFrom(state.type.stateType.arrayType)
    for arg in function.inputs:
        if arg.name in states:
            arg.type.CopyFrom(deepcopy(arg.type.stateType.wrappedType))
    del spec.description.state[:]

    def constant(name: str, values: list, boolean: bool = False) -> Any:
        op = mil.Operation(type="const")
        value = op.attributes["val"]
        value.type.tensorType.dataType = mil.BOOL if boolean else mil.INT32
        value.type.tensorType.rank = 1
        value.type.tensorType.dimensions.add().constant.size = len(values)
        if boolean:
            value.immediateValue.tensor.bools.values.extend(values)
        else:
            value.immediateValue.tensor.ints.values.extend(values)
        op.outputs.add(name=name).type.CopyFrom(value.type)
        return op

    block = next(iter(function.block_specializations.values()))
    if any(
        op.blocks or ("state" in op.type and op.type != "read_state")
        for op in block.operations
    ):
        raise ValueError(
            "Unsupported cache graph; expected read-only states without nested blocks"
        )
    aliases = {
        op.outputs[0].name: op.inputs["input"].arguments[0].name
        for op in block.operations
        if op.type == "read_state"
    }
    ops, mapped = [], set()
    for original in block.operations:
        if original.type == "read_state":
            continue
        op = deepcopy(original)
        for binding in op.inputs.values():
            for arg in binding.arguments:
                if arg.name in aliases:
                    arg.name = aliases[arg.name]
        ops.append(op)
        if op.type != "concat":
            continue
        inputs = [arg.name for arg in op.inputs["values"].arguments]
        cache_names = [name for name in inputs if name in states]
        if not cache_names:
            continue
        if len(cache_names) != 1 or inputs[0] != cache_names[0]:
            raise ValueError(
                "Expected one cache prepended to each causal convolution input"
            )
        name = cache_names[0]
        shape = list(states[name].type.stateType.arrayType.shape)
        full_shape = [d.constant.size for d in op.outputs[0].type.tensorType.dimensions]
        if (
            name in mapped
            or len(shape) != 3
            or shape[:2] != full_shape[:2]
            or not 0 < shape[2] < full_shape[2]
        ):
            raise ValueError(f"Unsupported convolution cache shape: {name}")
        output_name = "updated_" + name
        kwargs = {
            "begin": [0, 0, full_shape[2] - shape[2]],
            "end": full_shape,
            "stride": [1, 1, 1],
            "begin_mask": [False] * 3,
            "end_mask": [False] * 3,
            "squeeze_mask": [False] * 3,
        }
        slicing = mil.Operation(type="slice_by_index")
        slicing.inputs["x"].arguments.add().name = op.outputs[0].name
        for key, value in kwargs.items():
            const_name = output_name + "_" + key
            ops.append(constant(const_name, value, boolean=key.endswith("mask")))
            slicing.inputs[key].arguments.add().name = const_name
        slicing.outputs.add(name=output_name).type.CopyFrom(
            next(arg.type for arg in function.inputs if arg.name == name)
        )
        ops.append(slicing)
        block.outputs.append(output_name)
        spec.description.output.add(name=output_name).type.multiArrayType.CopyFrom(
            states[name].type.stateType.arrayType
        )
        mapped.add(name)
    if mapped != set(states):
        raise ValueError("Not every semantic cache has a corresponding causal concat")
    del block.operations[:]
    block.operations.extend(ops)
    return spec


def prepare_explicit_cache_model(source: Path, cache_dir: Path) -> Path:
    """Cache a converted package keyed by source contents and converter version."""
    import coremltools as ct

    component = source / "Data" / "com.apple.CoreML"
    fingerprint = hashlib.sha256(b"vibevoice-explicit-caches-v1")
    for path in sorted(p for p in component.rglob("*") if p.is_file()):
        fingerprint.update(str(path.relative_to(component)).encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                fingerprint.update(chunk)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"semantic_explicit_{fingerprint.hexdigest()[:16]}.mlpackage"
    if target.exists():
        return target
    spec = explicit_cache_spec(ct.utils.load_spec(str(component / "model.mlmodel")))
    model = ct.models.MLModel(
        spec, weights_dir=str(component / "weights"), skip_model_load=True
    )
    with tempfile.TemporaryDirectory(dir=cache_dir) as directory:
        staged = Path(directory) / "encoder.mlpackage"
        model.save(str(staged))
        # Another process may have finished the same content-addressed export.
        if not target.exists():
            try:
                staged.rename(target)
            except OSError:
                if not target.exists():
                    raise
    return target


class ExplicitCacheEncoder:
    """Own the recurrent cache tensors for one semantic encoder instance."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.shapes = {
            feature.name: tuple(feature.type.multiArrayType.shape)
            for feature in model.get_spec().description.input
            if feature.name != "audio"
        }
        self.caches: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.caches = {
            name: np.zeros(shape, dtype=np.float16)
            for name, shape in self.shapes.items()
        }

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        result = self.model.predict({"audio": audio, **self.caches})
        self.caches = {name: result["updated_" + name] for name in self.shapes}
        return result["features"]
