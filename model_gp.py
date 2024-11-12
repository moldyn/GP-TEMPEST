# model.py without GNNs
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_gp import TEMPEST


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
        kernel,
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
        self.kernel = kernel
        self.tempest = TEMPEST(
            cuda=self.cuda,
            dim_input=self.dim_input,
            dim_latent=self.dim_latent,
            layers_hidden_encoder=self.layers_hidden_encoder,
            layers_hidden_decoder=self.layers_hidden_decoder,
            inducing_points=self.inducing_points,
        )

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=self.cuda)
        self.load_state_dict({
            k: v for k, v in pretrained_dict.items() if k in self.state_dict()
        })

    def forward(self, x, t):
        """Forward pass through the network."""
        self.train()
        qzx = self.encoder(x)
        qzx_mu = qzx['means']
        qzx_var = qzx['variances']




