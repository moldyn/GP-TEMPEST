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