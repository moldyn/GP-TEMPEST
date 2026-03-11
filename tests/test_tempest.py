"""Tests for the TEMPEST model."""

import numpy as np
import pytest
import torch

from tempest_fc import TEMPEST, MaternKernel


@pytest.fixture
def small_model():
    kernel = MaternKernel(scale=10.0, nu=2.5, dtype=torch.float64)
    inducing_points = np.linspace(0, 1, 5)
    return TEMPEST(
        cuda=False,
        kernel=kernel,
        dim_input=4,
        dim_latent=2,
        layers_hidden_encoder=[16, 8],
        layers_hidden_decoder=[8, 16],
        inducing_points=inducing_points,
        beta=1.0,
        N_data=100,
        dtype=torch.float64,
    )


def test_tempest_init(small_model):
    assert small_model is not None
    assert small_model.dim_latent == 2


def test_encoder_output_shape(small_model):
    x = torch.randn(8, 4, dtype=torch.float64)
    out = small_model.encoder(x)
    assert out["means"].shape == (8, 2)
    assert out["variances"].shape == (8, 2)
    assert out["embedding"].shape == (8, 2)


def test_encoder_variances_positive(small_model):
    x = torch.randn(16, 4, dtype=torch.float64)
    out = small_model.encoder(x)
    assert (out["variances"] > 0).all()


def test_extract_latent_space_shape(small_model):
    from torch.utils.data import TensorDataset

    n = 20
    x = torch.randn(n, 4, dtype=torch.float64)
    t = torch.linspace(0, 1, n, dtype=torch.float64).unsqueeze(-1)
    dataset = TensorDataset(x, t)
    embedding = small_model.extract_latent_space(dataset, batch_size=8)
    assert embedding.shape == (n, 2)
