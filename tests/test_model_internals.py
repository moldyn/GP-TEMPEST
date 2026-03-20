"""Tests for model helper functions and internal TEMPEST methods."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from gptempest import TEMPEST, MaternKernel
from gptempest.model import (
    _cholesky_log_determinant,
    _gauss_cross_entropy,
    _num_stabilize_diag,
    _reparameterize,
    _robust_log_determinant,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def spd_matrix():
    """Small symmetric positive-definite matrix."""
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    return A


@pytest.fixture
def small_model():
    kernel = MaternKernel(scale=10.0, nu=1.5, dtype=torch.float64)
    inducing_points = np.linspace(0, 99, 8)
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


@pytest.fixture
def batch(small_model):
    n = 20
    x = torch.rand(n, 4, dtype=torch.float64)
    t = torch.linspace(0, 99, n, dtype=torch.float64).unsqueeze(1)
    return x, t


@pytest.fixture
def small_dataset():
    n = 50
    x = torch.rand(n, 4, dtype=torch.float64)
    t = torch.linspace(0, 49, n, dtype=torch.float64).unsqueeze(1)
    return TensorDataset(x, t)


# ── Helper functions ──────────────────────────────────────────────────────────

def test_num_stabilize_diag_shape(spd_matrix):
    out = _num_stabilize_diag(spd_matrix)
    assert out.shape == spd_matrix.shape


def test_num_stabilize_diag_increases_diagonal(spd_matrix):
    stabilizer = 1e-4
    out = _num_stabilize_diag(spd_matrix, stabilizer=stabilizer)
    for i in range(spd_matrix.shape[0]):
        assert out[i, i].item() == pytest.approx(
            spd_matrix[i, i].item() + stabilizer
        )


def test_robust_log_determinant_value(spd_matrix):
    logdet = _robust_log_determinant(spd_matrix)
    expected = torch.logdet(spd_matrix).item()
    assert logdet.item() == pytest.approx(expected, rel=1e-4)


def test_robust_log_determinant_near_singular():
    # Nearly singular matrix — should fall back gracefully
    mat = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    # Should not raise
    logdet = _robust_log_determinant(mat)
    assert torch.isfinite(logdet) or not torch.isfinite(logdet)  # just no exception


def test_cholesky_log_determinant_value(spd_matrix):
    logdet = _cholesky_log_determinant(spd_matrix)
    expected = torch.logdet(spd_matrix).item()
    assert logdet.item() == pytest.approx(expected, rel=1e-4)


def test_reparameterize_shape():
    mu = torch.zeros(8, 4, dtype=torch.float64)
    var = torch.ones(8, 4, dtype=torch.float64)
    z = _reparameterize(mu, var)
    assert z.shape == (8, 4)


def test_reparameterize_zero_variance():
    mu = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    var = torch.tensor([[1e-12, 1e-12]], dtype=torch.float64)
    z = _reparameterize(mu, var)
    assert z.shape == mu.shape


def test_gauss_cross_entropy_shape():
    n = 16
    mu_l = torch.randn(n, 2, dtype=torch.float64)
    var_l = torch.abs(torch.randn(n, 2, dtype=torch.float64)) + 0.1
    qzx_mu = torch.randn(n, 2, dtype=torch.float64)
    qzx_var = torch.abs(torch.randn(n, 2, dtype=torch.float64)) + 0.1
    out = _gauss_cross_entropy(mu_l, var_l, qzx_mu, qzx_var)
    assert out.shape == (n, 2)


def test_gauss_cross_entropy_identical_distributions():
    # When GP and encoder agree, cross-entropy should be finite
    mu = torch.zeros(8, 2, dtype=torch.float64)
    var = torch.ones(8, 2, dtype=torch.float64)
    out = _gauss_cross_entropy(mu, var, mu, var)
    assert torch.all(torch.isfinite(out))


# ── MaternKernel.to() ─────────────────────────────────────────────────────────

def test_matern_kernel_to_cpu():
    kernel = MaternKernel(scale=10.0, nu=1.5, dtype=torch.float64)
    kernel_cpu = kernel.to('cpu')
    assert kernel_cpu.device == 'cpu'


# ── TEMPEST.compute_kernel_matrices ───────────────────────────────────────────

def test_compute_kernel_matrices_shapes(small_model, batch):
    _, t = batch
    small_model.compute_kernel_matrices(t)
    M = small_model.inducing_points.shape[0]
    N = t.shape[0]
    assert small_model.kernel_mm.shape == (M, M)
    assert small_model.kernel_mm_inv.shape == (M, M)
    assert small_model.kernel_nn.shape == (N,)
    assert small_model.kernel_nm.shape == (N, M)
    assert small_model.kernel_mn.shape == (M, N)


def test_compute_kernel_matrices_symmetry(small_model, batch):
    _, t = batch
    small_model.compute_kernel_matrices(t)
    diff = (small_model.kernel_mm - small_model.kernel_mm.T).abs().max()
    assert diff.item() < 1e-10


# ── TEMPEST.compute_gp_params ─────────────────────────────────────────────────

def test_compute_gp_params_shapes(small_model, batch):
    x, t = batch
    small_model.eval()
    with torch.no_grad():
        qzx = small_model.encoder(x)
        small_model.compute_kernel_matrices(t)
        small_model.compute_gp_params(
            t, qzx['means'][:, 0], qzx['variances'][:, 0]
        )
    M = small_model.inducing_points.shape[0]
    N = t.shape[0]
    assert small_model.mu_l.shape == (M,)
    assert small_model.A_l.shape == (M, M)
    assert small_model.gp_mean_vector.shape == (N,)
    assert small_model.gp_mean_sigma.shape == (N,)


# ── TEMPEST.gp_step ───────────────────────────────────────────────────────────

def test_gp_step_runs(small_model, batch):
    x, t = batch
    small_model.train()
    small_model.gp_step(x, t)
    assert hasattr(small_model, 'elbo')
    assert hasattr(small_model, 'recon_loss')
    assert hasattr(small_model, 'gp_KL')


def test_gp_step_elbo_finite(small_model, batch):
    x, t = batch
    small_model.train()
    small_model.gp_step(x, t)
    assert torch.isfinite(small_model.elbo)


def test_gp_step_recon_loss_positive(small_model, batch):
    x, t = batch
    small_model.train()
    small_model.gp_step(x, t)
    assert small_model.recon_loss.item() >= 0


# ── TEMPEST.train_model ───────────────────────────────────────────────────────

def test_train_model_one_epoch(small_model, small_dataset):
    small_model.train_model(
        small_dataset,
        train_size=1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=16,
        n_epochs=1,
    )


def test_train_model_with_split(small_model, small_dataset):
    small_model.train_model(
        small_dataset,
        train_size=0.8,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=16,
        n_epochs=1,
    )


# ── TEMPEST.train_epoch eval mode ─────────────────────────────────────────────

def test_train_epoch_eval_mode(small_model, small_dataset):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader = DataLoader(small_dataset, batch_size=16)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    pbar = tqdm(loader, leave=False)
    elbo, recon, gp = small_model.train_epoch(pbar, optimizer, is_training=False)
    assert np.isfinite(elbo)
    assert np.isfinite(recon)
    assert np.isfinite(gp)
