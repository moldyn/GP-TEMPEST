# utils.py
# -*- coding: utf-8 -*-
"""Class with helper functions.

MIT License
Copyright (c) 2024, Georg Diez
All rights reserved.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import prettypyplot as pplt
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

pplt.use_style()


def create_deterministic_loader(batch_size, input_dim, num_samples, seed=42):
    torch.manual_seed(seed)
    features = torch.randn(num_samples, input_dim, dtype=torch.float64)
    times = torch.linspace(0, 1, num_samples, dtype=torch.float64).unsqueeze(-1)
    dataset = TensorDataset(features, times)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_prepare_data(input, dtype):
    """Load and prepare the data. Returns a TensorDataset."""
    scaler = MinMaxScaler()
    features = np.loadtxt(input)
    normalized_features = scaler.fit_transform(features)
    times = np.arange(len(features)).reshape(-1, 1)
    dataset = TensorDataset(
        torch.tensor(normalized_features, dtype=dtype),
        torch.tensor(times, dtype=dtype),
    )
    return dataset



def plot_distribution(distances, savename):
    pplt.update_style(style='minimal')
    distances = np.loadtxt(distances)
    hist, bins = np.histogram(distances, bins=50, density=True)
    _, ax = plt.subplots(figsize=(2.8, 1.75))
    ax.stairs(hist, bins, fill=True)
    ax.axvline(1.05, color='pplt:red', lw=1.0, label=r'$d_{\rm cut}$')
    ax.set_xlabel(r'$d_{ij}$')
    ax.set_ylabel(r'$p \left( d_{ij} \right)$')
    ax.set_xlim(0, 2)
    pplt.legend()
    pplt.savefig(savename)
    pplt.update_style(style='default')


def plot_latent_space(embedding, filename):
    """Plot the latent space."""
    hist, xedges, yedges = np.histogram2d(
        embedding.T[0],
        embedding.T[1],
        density=True,
        bins=50,
    )
    hist = hist.T
    xedges = 0.5 * (xedges[1:] + xedges[:-1])
    yedges = 0.5 * (yedges[1:] + yedges[:-1])

    deltag = -np.log(hist)
    deltag -= np.nanmin(deltag)
    deltag[deltag == np.nanmax(deltag)] = None

    _, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(2, 2),
        gridspec_kw={
            'wspace': 0.3,
        },
    )
    axs[0].contourf(
        xedges,
        yedges,
        deltag,
    )
    axs[1].scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        ec=None,
        s=1,
        alpha=0.9,
        cmap='tab10',
    )
    for ax in axs:
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
    pplt.savefig(filename)


def yaml_config_reader(config: str):
    """Parse all parameters using the yaml file."""
    with open(config, 'r') as stream:
        params = yaml.safe_load(stream)

    # Create header string with all parameters for reproducibility
    header = (
        "# TEMPEST model configuration Parameters:\n"
        f"# data_path: {params.get('data_path')}\n"
        f"# inducing_points_path: {params.get('inducing_points_path')}\n"
        f"# save_path: {params.get('save_path')}\n"
        f"# cuda: {params.get('cuda')}\n"
        f"# dim_input: {params.get('dim_input')}\n"
        f"# dim_latent: {params.get('dim_latent')}\n"
        f"# neurons_ae: {params.get('neurons_ae')}\n"
        f"# epochs: {params.get('epochs')}\n"
        f"# batch_size: {params.get('batch_size')}\n"
        f"# learning_rate: {params.get('learning_rate')}\n"
        f"# weight_decay: {params.get('weight_decay')}\n"
        f"# beta: {params.get('beta')}\n"
        f"# kernel_nu: {params.get('kernel_nu')}\n"
        f"# kernel_scale: {params.get('kernel_scale')}"
    )
    return (
        params.get('data_path'),
        params.get('inducing_points_path'),
        params.get('save_path'),
        params.get('cuda'),
        int(params.get('dim_input')),
        int(params.get('dim_latent')),
        list(params.get('neurons_ae')),
        int(params.get('epochs')),
        int(params.get('batch_size')),
        float(params.get('learning_rate')),
        float(params.get('weight_decay')),
        float(params.get('beta')),
        float(params.get('kernel_nu')),
        float(params.get('kernel_scale')),
        header,
    )



def generate_yaml_config(output_file):
    """Generates a YAML config file"""
    base_name, ext = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(output_file):
        output_file = f'{base_name}_{counter}{ext}'
        counter += 1

    config_data = {
        'data_path': 'path/to/default/data/file',
        'inducing_points_path': 'path/to/default/inducing_points/file',
        'save_path': 'path/to/output/files/',
        'cuda': True,
        'dim_input': 2,
        'dim_latent': 2,
        'neurons_ae': [32, 32, 32],
        'epochs': 100,
        'batch_size': 1024,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'beta': 50,
        'kernel_nu': 1.5,
        'kernel_scale': 1e3,
    }
    comments = {
        'data_path': 'File path containing the coordinates.',
        'inducing_points_path': 'File path containing the timestamps of the inducing points.',
        'save_path': 'File path which will contain all output files.',
        'cuda': 'If set, the training runs on GPUs (they must be CUDA-compatible)',
        'dim_input': 'Dimensionality of the input layer (number of features)',
        'dim_latent': 'Dimensionality of the latent space',
        'neurons_ae': 'The dimensions of the hidden layers of the autoencoder [q(z|x)].',
        'epochs': 'Number of epochs for the model to train.',
        'batch_size': 'Batch size: number of samples passed through the network at a time',
        'learning_rate': 'Learning rate (usually 1e-2 - 1e-6)',
        'weight_decay': 'The decay rate for the optimizer',
        'beta': 'Weight of the Gaussian process loss term',
        'kernel_nu': 'The parameter nu in the Matern kernel',
        'kernel_scale': 'The scale parameter in the Matern kernel (time scale)',
    }
    with open(output_file, 'w') as yaml_file:
        for key, value in config_data.items():
            if key in comments:
                yaml_file.write(f"# {comments[key]}\n")
            yaml.dump({key: value}, yaml_file, default_flow_style=False)
    return output_file



