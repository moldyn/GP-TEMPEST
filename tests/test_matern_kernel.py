"""Tests for MaternKernel."""

import pytest
import torch

from tempest_fc import MaternKernel


@pytest.fixture
def t1():
    return torch.linspace(0, 1, 10, dtype=torch.float64).unsqueeze(-1)


@pytest.fixture
def t2():
    return torch.linspace(0, 1, 5, dtype=torch.float64).unsqueeze(-1)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernel_init(nu):
    kernel = MaternKernel(scale=1.0, nu=nu, dtype=torch.float64)
    assert kernel.nu == nu


def test_matern_kernel_invalid_nu():
    with pytest.raises(RuntimeError):
        MaternKernel(scale=1.0, nu=1.0, dtype=torch.float64)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_kernel_mat_shape(nu, t1, t2):
    kernel = MaternKernel(scale=1.0, nu=nu, dtype=torch.float64)
    K = kernel.kernel_mat(t1, t2)
    assert K.shape == (t1.shape[0], t2.shape[0])


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_kernel_mat_values_in_range(nu, t1):
    kernel = MaternKernel(scale=1.0, nu=nu, dtype=torch.float64)
    K = kernel.kernel_mat(t1, t1)
    assert (K >= 0).all(), "Kernel values should be non-negative"
    assert (K <= 1.0 + 1e-6).all(), "Kernel values should be <= 1"


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_kernel_diag_shape(nu, t1):
    kernel = MaternKernel(scale=1.0, nu=nu, dtype=torch.float64)
    diag = kernel.kernel_diag(t1, t1)
    assert diag.shape == (t1.shape[0],)
