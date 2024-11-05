# model.py without GNNs
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_stabilize_diag(mat, stabilizer=1e-8):
    diag = torch.eye(mat.size(-1), device=mat.device).expand(mat.shape)
    return mat + stabilizer * diag


def _cholesky_log_determinant(mat):
    cholesky_decomposition = torch.linalg.cholesky(
        _num_stabilize_diag(mat),
    )
    return 2 * torch.sum(torch.log(torch.diagonal(cholesky_decomposition)))





class FeedForwardNN(nn.Module):
    """Simple linear NN with ReLU activation."""
    def __init__(self, layer_sizes):
        super().__init__()
        # Check if layer_sizes is a list of integers
        assert isinstance(layer_sizes, list), (
            'layer_sizes must be a list'
        )
        assert all(isinstance(size, int) for size in layer_sizes), (
            'All elements in layer_sizes must be integers'
        )
        layers = []
        for layer_nr in range(len(layer_sizes) - 1):
            layers.append(
                nn.Linear(layer_sizes[layer_nr], layer_sizes[layer_nr + 1]),
            )
            if layer_nr < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.ModuleList(layers)

    def add_layer(self, name, layer):
        """Add a layer or some layers to the model."""
        assert hasattr(self, 'model'), ('The model is not yet initialized.')
        self.model.add_module(f'{name}', layer)

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.model:
            x = layer(x)
        return x


class GaussianLayer(nn.Module):
    """Class for Gaussian Sampling.

    Parameters
    ----------
        dim_input: int
            Integer defining the input dimensionality before the last layer

        dim_latent : int
            Integer defining the dimension of the latent space

    Returns
    -------
        mu : array
            learned means of the samples in the latent space

        var : array
            the corresponding log variance of the Gaussian distributions

        z : array
            sampled points from the learned distribution

    """

    def __init__(self, layers_gaussian):
        """Initialize Gaussian layer class."""
        super().__init__()
        self.mu = FeedForwardNN(layers_gaussian)
        self.var = FeedForwardNN(layers_gaussian)

    def reparameterize(self, mu, var):
        """Reparameterize to enable backpropagation."""
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Learns latent space and samples from learned Gaussian."""
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class InferenceNN(nn.Module):
    """The inference network corresponds to the encoder in a VAE."""
    def __init__(
        self,
        layers_encoder,
    ):
        super().__init__()
        self.inference_qzx = FeedForwardNN(layers_encoder)
        self.inference_qzx.add_layer('RELU', nn.ReLU())
        self.inference_qzx.add_layer(
            'GaussianLayer', GaussianLayer(layers_encoder),
        )

    def forward(self, x):
        """Compute VAE latent space."""
        mu, var, z = self.inference_qzx(x)
        return {
            'means': mu,
            'variances': var,
            'embedding': z,
        }


class MaternKernel(nn.Module):
    def __init__(self, scale=1, nu=1.5, device='cuda'):
        super(MaternKernel, self).__init__()
        self.scale = torch.tensor([scale], dtype=torch.float32).to(device)
        self.nu = nu

        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError('nu expected to be 0.5, 1.5, or 2.5')

    def _compute_kernel(self, distance):
        exp_component = torch.exp(-torch.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            prefac = 1
        elif self.nu == 1.5:
            prefac = (torch.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            prefac = (torch.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return prefac * exp_component

    def kernel_mat(self, t1, t2):
        mean = t1.mean(dim=-2, keepdim=True)
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        distance = torch.cdist(t1_s, t2_s).clamp(min=1e-15)
        return self._compute_kernel(distance)

    def kernel_diag(self, t1, t2):
        mean = t1.mean(dim=-2, keepdim=True)
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        distance = ((t1_s - t2_s)**2).sum(dim=1).sqrt().clamp(min=1e-15)
        return self._compute_kernel(distance)


class TEMPEST(nn.Module):
    def __init__(
        self,
        cuda,
        kernel,
        dim_input,
        dim_latent,
        layers_hidden_encoder,
        layers_hidden_decoder,
        inducing_points,
        N_data,
    ):
        """Initializes the TEMPEST network architecture."""
        super().__init__()
        self.cuda = cuda
        self.kernel = kernel
        self.inducing_points = inducing_points
        self.layers_encoder = [dim_input, *layers_hidden_encoder, dim_latent]
        self.layers_decoder = [dim_latent, *layers_hidden_decoder, dim_input]
        self.encoder = InferenceNN(self.layers_encoder)
        self.decoder = FeedForwardNN(self.layers_decoder)
        self.N_data = N_data

    def compute_kernel_matrices(self, t):
        self.kernel_mm = self.kernel.kernel_mat(
            self.inducing_points,
            self.inducing_points,
        )
        self.kernel_mm_inv = torch.linalg.inv(
            _num_stabilize_diag(self.kernel_mm),
        )
        self.kernel_nn = self.kernel.kernel_diag(t, t)
        self.kernel_nm = self.kernel.kernel_mat(t, self.inducing_points)
        self.kernel_mn = torch.transpose(self.kernel_nm, 0, 1)

    def _compute_diagonal_kernel(self, precision):
        """Compute the diagonal elements of the kernel matrix."""
        return precision * (
            self.kernel_nn - torch.diagonal(
                torch.matmul(
                    self.kernel_nm,
                    torch.matmul(
                        self.kernel_mm_inv,
                        self.kernel_mn,
                    ),
                ),
            ),
        )

    def _compute_Lambda(self):
        """Compute the Lambda matrix."""
        return torch.matmul(
            self.kernel_mm_inv,
            torch.matmul(
                torch.matmul(
                    self.kernel_nm.unsqueeze(2),
                    torch.transpose(self.kernel_nm.unsqueeze(2), 1, 2),
                ),
                self.kernel_mm_inv,
            ),
        )

    def approximate_posterior(self, t, qzx_mu, qzx_var):
        constant = self.N_data / t.shape[0]
        Sigma_l = self.kernel_mm + constant * torch.matmul(
            self.kernel_mn,
            self.kernel_nm / qzx_var.unsqueeze(1),
        )  # see Eq.(9) in Jazbec21
        Sigma_l_inv = torch.linalg.inv(_num_stabilize_diag(Sigma_l))
        self.mu_l = constant * torch.matmul(
            self.kernel_mm,  # error: original code takes self.kernel_nm
            torch.matmul(
                Sigma_l_inv,
                torch.matmul(
                    self.kernel_mn,
                    qzx_mu / qzx_var,
                )
            )
        )
        self.A_l = torch.matmul(
            self.kernel_mm,
            torch.matmul(
                Sigma_l_inv,
                self.kernel_mm,
            ),
        )
        self.GP_mean_vector = constant * torch.matmul(
            self.kernel_nm,
            torch.matmul(
                Sigma_l_inv,
                torch.matmul(
                    self.kernel_mn,
                    qzx_mu / qzx_var,
                )
            )
        )  # Eq.(7) in Tian24 Methods
        self.GP_mean_sigma = self.kernel_nn + torch.diagonal(
            -torch.matmul(
                self.kernel_nm,
                torch.matmul(
                    self.kernel_mm_inv,
                    self.kernel_mn,
                )
            ) + torch.matmul(
                self.kernel_nm,
                torch.matmul(
                    Sigma_l_inv,
                    self.kernel_mn,
                )
            )
        )
        # GP_mean_vector and GP_mean_sigma is updated posterior, while mu_l and A_l acts more like a prior (compare eq 9 in Tian24)


    def variational_loss(self, qzx_mu, qzx_var):
        """
        Compute the Hensman loss term for the current batch.
        Compare eg. Eq.(7) and (10) in Jazbec21.
        More details in Jazbec21 SI B, proposition B.1
        """
        m = self.inducing_points.shape[0]
        log_det_kmm = _cholesky_log_determinant(self.kernel_mm)
        log_det_A = _cholesky_log_determinant
        KL_div = 0.5 * (-m + torch.trace(torch.matmul(
            self.kernel_mm_inv,
            self.A_l,
        )) + torch.sum(self.mu_l * torch.matmul(
            self.kernel_mm_inv,
            self.mu_l,
        ) + log_det_kmm - log_det_A))
        # compute L3 sum term
        mean_vec = torch.matmul(
            self.kernel_mn,
            torch.matmul(
                self.kernel_mm_inv,
                self.mu_l,
            )
        )  # first term in Jazbec21 SI, B.1 first eq.
        precision = 1 / qzx_var
        k_iitilde = self._compute_diagonal_kernel(precision)
        Lambda = self._compute_Lambda()
        tr_ALambda = precision * torch.einsum(
            'bii->b',
            torch.matmul(
                self.A_l,
                Lambda,
            ),
        )
        loss_term_L3 = -0.5 * torch.sum(
            torch.sum(k_iitilde) + torch.sum(tr_ALambda) +
            torch.sum(torch.log(qzx_var)) + m *
            torch.log(2 * 3.1415927410125732) +
            torch.sum(precision * (qzx_mu - mean_vec)**2)
        )
        return loss_term_L3, KL_div

    def gauss_cross_entropy(mu_l, GP_mean_sigma, qzx_mu, qzx_var):
        """Proof see SI Tian24, Proposition 4 on p.83"""
        log2pi = 1.8378770664093453
        log_qzx_var = torch.log(qzx_var)
        scaled_square_diff = (
            GP_mean_sigma + mu_l**2 - 2 * mu_l * qzx_mu + qzx_mu**2
        ) / qzx_var
        return - 0.5 * (log2pi + log_qzx_var + scaled_square_diff)

    def gp_step(self, x, t):
        qzx = self.encoder(x)
        qzx_mu = qzx['means']
        qzx_var = qzx['variances']

        self.compute_kernel_matrices(t)
        for latent_dim in range(self.dim_latent):  # l for channel
            self.approximate_posterior(
                t,
                qzx_mu[latent_dim],  # check dim of qzx_mu
                qzx_var[latent_dim],
            )
            self.variational_loss(
                qzx_mu[latent_dim],  # check dim of qzx_mu
                qzx_var[latent_dim],
            )





        x_recon = self.decoder(z)
        return x_recon

