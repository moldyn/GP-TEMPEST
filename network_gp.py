# model.py without GNNs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, random_split


def _num_stabilize_diag(mat, stabilizer=1e-8):
    diag = torch.eye(mat.size(-1), device=mat.device).expand(mat.shape)
    return mat + stabilizer * diag


def _cholesky_log_determinant(mat):
    cholesky_decomposition = torch.linalg.cholesky(
        _num_stabilize_diag(mat),
    )
    return 2 * torch.sum(torch.log(torch.diagonal(cholesky_decomposition)))


def _reparameterize(mu, var):
    """Reparameterize to enable backpropagation."""
    std = torch.exp(0.5 * torch.log(var))
    eps = torch.randn_like(std)
    return mu + eps * std


def _gauss_cross_entropy(mu_l, GP_mean_sigma, qzx_mu, qzx_var):
    """Proof see SI Tian24, Proposition 4 on p.83"""
    log2pi = 1.8378770664093453
    log_qzx_var = torch.log(qzx_var)
    scaled_square_diff = (
        GP_mean_sigma + mu_l**2 - 2 * mu_l * qzx_mu + qzx_mu**2
    ) / qzx_var
    return - 0.5 * (log2pi + log_qzx_var + scaled_square_diff)

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

    def forward(self, x):
        """Learns latent space and samples from learned Gaussian."""
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = _reparameterize(mu, var)
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
    def __init__(self, scale=1, nu=1.5):
        super(MaternKernel, self).__init__()
        self.scale = torch.tensor([scale], dtype=torch.float32)
        self.device = self.scale.device
        self.dtype = self.scale.dtype
        self.nu = nu
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError('nu expected to be 0.5, 1.5, or 2.5')

    def to(self, device):
        """Move the kernel and its parameters to the specified device."""
        self.device = device
        self.scale = self.scale.to(device)
        return self

    def _compute_kernel(self, distance):
        exp_component = torch.exp(-torch.sqrt(torch.tensor(
            self.nu * 2, dtype=self.dtype, device=self.device
        )) * distance)
        if self.nu == 0.5:
            prefac = 1
        elif self.nu == 1.5:
            prefac = (torch.sqrt(
                torch.tensor(3.0, dtype=self.dtype, device=self.device)
            ) * distance).add(1)
        elif self.nu == 2.5:
            prefac = (torch.sqrt(
                torch.tensor(5.0, dtype=self.dtype, device=self.device)
            ) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return prefac * exp_component

    def kernel_mat(self, t1, t2):
        t1 = t1.clone().detach().to(self.device).unsqueeze(-1)
        t2 = t2.clone().detach().to(self.device).unsqueeze(-1)
        mean = t1.mean()
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        distance = torch.cdist(t1_s, t2_s).clamp(min=1e-15)
        return self._compute_kernel(distance)

    def kernel_diag(self, t1, t2):
        t1 = t1.clone().detach().to(self.device)
        t2 = t2.clone().detach().to(self.device)
        mean = t1.mean()
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
        beta,
        N_data,
    ):
        """Initializes the TEMPEST network architecture.
            inducing_points (array_like): the inducing points which are used
                for the sparse GP regression. This inducing points should cover
                important events over the time series such as transitions and
                timepoints in which the system is in a metastable state.
        """
        super().__init__()
        self.dtype = torch.float32
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.kernel = kernel.to(self.device)
        self.inducing_points = torch.tensor(inducing_points, dtype=self.dtype).to(self.device)
        self.dim_latent = dim_latent
        self.layers_encoder = [dim_input, *layers_hidden_encoder, dim_latent]
        self.layers_decoder = [dim_latent, *layers_hidden_decoder, dim_input]
        self.encoder = InferenceNN(self.layers_encoder).to(self.device)
        self.decoder = FeedForwardNN(self.layers_decoder).to(self.device)
        self.decoder.add_layer('sigmoid', nn.Sigmoid())
        self.beta = beta
        self.N_data = N_data

    def compute_kernel_matrices(self, t):
        self.kernel_mm = self.kernel.kernel_mat(
            self.inducing_points,
            self.inducing_points,
        )
        print(self.kernel_mm.device)
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

    def compute_gp_params(self, t, qzx_mu, qzx_var):
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
        log_det_A = _cholesky_log_determinant(self.A_l)
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
        loss_recon = -0.5 * torch.sum(
            torch.sum(k_iitilde) + torch.sum(tr_ALambda) +
            torch.sum(torch.log(qzx_var)) + m *
            torch.log(2 * 3.1415927410125732) +
            torch.sum(precision * (qzx_mu - mean_vec)**2)
        )  # this is the L3 loss from Hensman
        return loss_recon, KL_div

    def gp_step(self, x, t):
        print(x.shape)
        qzx = self.encoder(x)
        qzx_mu = qzx['means']
        qzx_var = qzx['variances']
        self.compute_kernel_matrices(t)
        gp_mean, gp_var = [], []
        loss_recon, loss_KL = [], []
        for latent_dim in range(self.dim_latent):  # l for channel
            self.compute_gp_params(
                t,
                qzx_mu[latent_dim],  # check dim of qzx_mu
                qzx_var[latent_dim],
            )
            gp_mean.append(self.GP_mean_vector)
            gp_var.append(self.GP_mean_sigma)
            l_recon, l_KL = self.variational_loss(
                qzx_mu[latent_dim],  # check dim of qzx_mu
                qzx_var[latent_dim],
            )
            loss_recon.append(l_recon)
            loss_KL.append(l_KL)
        loss_recon = torch.sum(torch.stack(loss_recon, dim=-1))
        loss_KL = torch.sum(torch.stack(loss_KL, dim=-1))
        elbo_gp = loss_recon - (x.shape[0] / len(self.inducing_points)) * loss_KL
        gp_mean = torch.stack(gp_mean, dim=1)
        gp_var = torch.stack(gp_var, dim=1)
        gp_cross_entropy = torch.sum(
            _gauss_cross_entropy(gp_mean, gp_var, qzx_mu, qzx_var)
        )
        self.gp_KL = gp_cross_entropy - elbo_gp
        latent_dist = Normal(qzx_mu, torch.sqrt(qzx_var))  # todo: sqrt ja oder nein?
        latent_samples = latent_dist.rsample()

        # decode and reconstruction loss
        qxz = self.decoder(latent_samples)
        loss_L2 = nn.MSELoss(reduction='mean')
        self.recon_loss = loss_L2(qxz, x)
        self.elbo = self.recon_loss + self.beta * self.gp_KL

    def get_latent_space(self, x, t):
        self.eval()
        latent_samples = []

        print('if x not torch tensor then set it to torch tensor', type(x))  # if x not torch tensor then x = torch.tensor(x, dtype=self.dtype)
        print(x.shape[0], t.shape[0])

        num = t.shape[0]
        num_batches = int(num / self.batch_size)

        for idx_batch in range(num_batches):  # loop through all batches
            t_batch = t[
                idx_batch * self.batch_size:min(
                    (idx_batch + 1) * self.batch_size, num,
                )
            ].to(self.device)
            x_batch = x[
                idx_batch * self.batch_size:min(
                    (idx_batch + 1) * self.batch_size, num,
                )
            ].to(self.device)
            qzx = self.encoder(x_batch)
            qzx_mu = qzx['means']
            qzx_var = qzx['variances']  # maybe clamp to ensure positive variance?
            gp_mean, gp_var = [], []
            for latent_dim in range(self.dim_latent):
                self.compute_gp_params(
                    t_batch,
                    qzx_mu[latent_dim],
                    qzx_var[latent_dim],
                )
                gp_mean.append(self.GP_mean_vector)
                gp_var.append(self.GP_mean_sigma)
            gp_mean = torch.stack(gp_mean, dim=1)
            gp_var = torch.stack(gp_var, dim=1)
            latent_samples_batch = _reparameterize(gp_mean, gp_var)
            latent_samples.append(latent_samples_batch.cpu().detach().numpy())
        return torch.cat(latent_samples, dim=0)

    def train_model(
        self,
        dataset,
        train_size,
        learning_rate,
        weight_decay,
        batch_size,
        n_epochs,
    ):
        """Train the TEMPEST model.

        Args:
            learning_rate (float): learning rate; typicalle 1e-3 to 1e-5
            batch_size (int): the batch size. As the estimators for the sparse
                GP regression converge to the estimators for
                batch size -> number frames, we recommend larger
                batch sizes > 512
            n_epochs (int): number of epochs to train
        """
        train_dataset, test_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, 1 - train_size],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop_last=False,  # to do?
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(
           filter(lambda p: p.requires_grad, self.parameters()),
           lr=learning_rate,
           weight_decay=weight_decay,
        )
        for nr_epoch in range(n_epochs):
            l_train_elbo, l_train_recon, l_train_gp = self.train_epoch(
                train_loader, optimizer, is_training=True,
            )
            l_test_elbo, l_test_recon, l_test_gp = self.train_epoch(
                test_loader, is_training=False,
            )
            print(
                f'Epoch {nr_epoch}: ELBO | {l_train_elbo:.5f}, '
                f'Recon Loss {l_train_recon:.5f}, '
                f'GP Loss {l_train_gp:.5f} | ',
                f'Val ELBO | {l_test_elbo:.5f}, Val Recon {l_test_recon:.5f}, '
                f'Val GP Loss {l_test_gp:.5f}',
            )
        torch.save(self.state_dict(), 'model.pt')

    def train_epoch(self, loader, optimizer, is_training=True):
        """Train the model for one epoch.

        Args:
            loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.

        Returns:
            tuple: Average training losses (ELBO, reconstruction, GP KL) for the epoch.
        """
        nr_frames, loss_elbo, loss_recon, loss_gp = 0, 0, 0, 0

        for x_batch, t_batch in loader:
            x_batch = x_batch.clone().detach().to(self.device)
            t_batch = t_batch.clone().detach().to(self.device)
            # x_batch = torch.tensor(x_batch, dtype=self.dtype).to(self.device)  # use clone().detach() instead of torch.tensor but consumes more memory
            # t_batch = torch.tensor(t_batch, dtype=self.dtype).to(self.device)
            if is_training:
                optimizer.zero_grad()

            # Perform a step of GP computation and forward pass
            self.gp_step(x_batch, t_batch)

            # Accumulate losses
            loss_elbo += self.elbo.item()
            loss_recon += self.recon_loss.item()
            loss_gp += self.gp_KL.item()
            if is_training:
                self.elbo.backward()  # Backpropagation; removed retain_graph=True in order to save memory
                optimizer.step()  # Update model parameters
            nr_frames += t_batch.shape[0]

        return loss_elbo / nr_frames, loss_recon / nr_frames, \
            loss_gp / nr_frames

