"""Explicit Core ML caches must survive frames and reset between utterances."""

import numpy as np
import pytest

ct = pytest.importorskip("coremltools")
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

from vibevoice_mlx.coreml_semantic import ExplicitCacheEncoder, explicit_cache_spec


@pytest.fixture
def legacy_model():
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 1, 4), dtype=types.fp16),
            mb.StateTensorSpec(shape=(1, 1, 2), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def program(audio, cache):
        history = mb.read_state(input=cache)
        return mb.concat(values=[history, audio], axis=2, name="features")

    return ct.convert(
        program, minimum_deployment_target=ct.target.macOS15, skip_model_load=True
    )


def test_explicit_cache_history_and_reset(legacy_model) -> None:
    original = legacy_model.get_spec()
    spec = explicit_cache_spec(original)
    assert len(original.description.state) == 1
    assert not spec.description.state
    assert [f.name for f in spec.description.input] == ["audio", "cache"]
    model = ct.models.MLModel(
        spec,
        weights_dir=legacy_model.weights_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    encoder = ExplicitCacheEncoder(model)
    first = encoder(np.array([[[1, 2, 3, 4]]], dtype=np.float16))
    second = encoder(np.array([[[5, 6, 7, 8]]], dtype=np.float16))
    np.testing.assert_array_equal(first, [[[0, 0, 1, 2, 3, 4]]])
    np.testing.assert_array_equal(second, [[[3, 4, 5, 6, 7, 8]]])
    encoder.reset()
    np.testing.assert_array_equal(
        encoder(np.array([[[1, 2, 3, 4]]], dtype=np.float16)), first
    )


def test_rejects_unrecognized_state_updates(legacy_model) -> None:
    spec = legacy_model.get_spec()
    block = next(iter(spec.mlProgram.functions["main"].block_specializations.values()))
    block.operations.add(type="write_state")
    with pytest.raises(ValueError, match="Unsupported cache graph"):
        explicit_cache_spec(spec)
