"""Tests for FeedForwardNN and GaussianLayer."""

import pytest
import torch

from gptempest import FeedForwardNN, GaussianLayer


class TestFeedForwardNN:
    def test_init_valid(self):
        net = FeedForwardNN([10, 32, 16, 2])
        assert net is not None

    def test_init_invalid_not_list(self):
        with pytest.raises(AssertionError):
            FeedForwardNN((10, 32, 2))

    def test_init_invalid_non_int(self):
        with pytest.raises(AssertionError):
            FeedForwardNN([10, 32.0, 2])

    def test_output_shape(self):
        net = FeedForwardNN([10, 32, 2])
        x = torch.randn(8, 10)
        out = net(x)
        assert out.shape == (8, 2)

    def test_linear_mode(self):
        net = FeedForwardNN([10, 2], linear=True)
        x = torch.randn(4, 10)
        out = net(x)
        assert out.shape == (4, 2)

    def test_add_layer(self):
        net = FeedForwardNN([10, 32, 2])
        net.add_layer("sigmoid", torch.nn.Sigmoid())
        x = torch.randn(4, 10)
        out = net(x)
        assert out.shape == (4, 2)
        assert (out >= 0).all() and (out <= 1).all()


class TestGaussianLayer:
    def test_output_shapes(self):
        layer = GaussianLayer([16, 4])
        x = torch.randn(8, 16)
        mu, var, z = layer(x)
        assert mu.shape == (8, 4)
        assert var.shape == (8, 4)
        assert z.shape == (8, 4)

    def test_variance_positive(self):
        layer = GaussianLayer([16, 4])
        x = torch.randn(32, 16)
        _, var, _ = layer(x)
        assert (var > 0).all(), "Variances must be strictly positive"
