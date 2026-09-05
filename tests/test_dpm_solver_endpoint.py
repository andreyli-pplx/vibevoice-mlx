"""The zero-noise ODE endpoint must return the last clean prediction."""

import importlib
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

generation = importlib.import_module("vibevoice_mlx.generate")


@pytest.mark.parametrize("num_steps", [1, 10, 14, 15, 20, 50])
def test_zero_noise_endpoint_returns_final_clean_prediction(num_steps: int) -> None:
    condition = mx.zeros((1, 2), dtype=mx.float16)
    previous_clean = mx.full((1, generation.VAE_DIM), 0.1, dtype=mx.float32)
    final_clean = mx.full((1, generation.VAE_DIM), 0.05, dtype=mx.float32)
    predictions = [previous_clean] * (num_steps - 1) + [final_clean]

    # Control denoiser predictions while exercising the real solver updates.
    # The changed final prediction exposes invalid second-order extrapolation.
    with patch.object(generation, "_dpm_denoise_step", side_effect=predictions):
        result = generation.dpm_solver_2m(
            None, condition, condition, cfg_scale=1.3, num_steps=num_steps, seed=42
        )

    np.testing.assert_allclose(np.array(result), np.array(final_clean), atol=1e-6)
