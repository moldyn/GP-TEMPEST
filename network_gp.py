# model.py without GNNs
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, t1, t2):
        mean = t1.mean(dim=-2, keepdim=True)
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        distance = torch.cdist(t1_s, t2_s).clamp(min=1e-15)
        return self._compute_kernel(distance)

    def forward_diag(self, t1, t2):
        mean = t1.mean(dim=-2, keepdim=True)
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        distance = ((t1_s - t2_s)**2).sum(dim=1).sqrt().clamp(min=1e-15)
        return self._compute_kernel(distance)


class TEMPEST(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_latent,
        layers_hidden_encoder,
        layers_hidden_decoder,
        cuda,
    ):
        """Initializes the TEMPEST network architecture."""
        super().__init__()
        self.cuda = cuda
        self.layers_encoder = [dim_input, *layers_hidden_encoder, dim_latent]
        self.layers_decoder = [dim_latent, *layers_hidden_decoder, dim_input]
        self.encoder = InferenceNN(self.layers_encoder)
        self.decoder = FeedForwardNN(self.layers_decoder)

    def forward(self, x, t):
        """Forward pass through the network."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

