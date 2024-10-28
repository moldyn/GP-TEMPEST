# model.py without GNNs
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_gp import TEMPEST


class MaternKernel(nn.Module):
    def __init__(self, scale=1, nu=1.5, device='cuda'):
        super(MaternKernel, self).__init__()
        self.scale = torch.tensor([scale], dtype=torch.float32).to(device)
        self.nu = nu

        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError('nu expected to be 0.5, 1.5, or 2.5')

    def forward(self, t1, t2):
        mean = t1.mean(dim=-2, keepdim=True)
        t1_s = (t1 - mean) / self.scale
        t2_s = (t2 - mean) / self.scale
        dist = torch.cdist(t1_s, t2_s).clamp(min=1e-15)
        if self.nu == 0.5:
            return torch.exp(-dist)
        elif self.nu == 1.5:
            sqrt3_dist = torch.sqrt(torch.tensor(3)) * dist
            return (1.0 + sqrt3_dist) * torch.exp(-sqrt3_dist)
        elif self.nu == 2.5:
            sqrt5_dist = torch.sqrt(torch.tensor(5)) * dist
            return (1.0 + sqrt5_dist + (5.0 / 3.0) * dist**2) * \
                torch.exp(-sqrt5_dist)
        else:
            raise RuntimeError('nu only implemented to be 0.5, 1.5, or 2.5')


class TEMPEST_model(nn.Module):
    def __init__(
        self,
        cuda,
        dim_input,
        dim_latent,
        layers_hidden_encoder,
        layers_hidden_decoder,
        epochs,
        batch_size,
        learning_rate,
        GP_w,
        GP_kernel,
    ):
        self.cuda = cuda
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.layers_hidden_encoder = layers_hidden_encoder
        self.layers_hidden_decoder = layers_hidden_decoder
        self.epoch = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.GP_w = GP_w
        self.GP_kernel = GP_kernel