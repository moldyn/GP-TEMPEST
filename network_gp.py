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


class InferenceNN(nn.Module):
    """The inference network corresponds to the encoder in a VAE."""
    def __init__(
        self,
        layers_encoder,
    ):
        super().__init__()


class TEMPEST(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_latent,
        layers_hidden_encoder,
        layers_hidden_decoder,
    ):
        """Initializes the TEMPEST network architecture."""
        super().__init__()
        self.layers_encoder = [dim_input, *layers_hidden_encoder, dim_latent]