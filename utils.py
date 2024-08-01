# utils.py
# -*- coding: utf-8 -*-
"""Class with helper functions.

MIT License
Copyright (c) 2022-2023, Georg Diez
All rights reserved.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import prettypyplot as pplt

pplt.use_style()


def gaussian_decay(d, a=1.2):
    return np.exp(- (d / a) ** 2)


def load_data(filename, contact_pairs, node_features_file):
    """Load and prepare data."""
    all_contact_distances = np.loadtxt(filename)
    max_edges = np.shape(all_contact_distances)[1]
    contact_pairs = np.loadtxt(contact_pairs) - 1
    node_features = np.loadtxt(node_features_file)
    node_features = node_features.reshape(-1, 162, 4)
    node_features = torch.tensor(node_features, dtype=torch.float)
    graphs = []
    one_hot_vector_list = []

    for num, contact_distances in enumerate(all_contact_distances):
        contact_idxs = np.where(contact_distances <= 1.05)[0]
        valid_contact_pairs = contact_pairs[contact_idxs]
        valid_contact_distances = contact_distances[contact_idxs]
        one_hot_vector = np.zeros(max_edges)
        one_hot_vector[contact_idxs] = 1
        one_hot_vector_list.append(one_hot_vector)

        edge_index = torch.tensor(
            valid_contact_pairs, dtype=torch.long,
        ).t().contiguous()
        edge_attr = torch.tensor(
            gaussian_decay(valid_contact_distances), dtype=torch.float,
            # valid_contact_distances, dtype=torch.float,
        ).view(-1, 1)

        graph = Data(
            x=node_features[num],
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        graphs.append(graph)
    return graphs, one_hot_vector_list



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




def plot_latent_space(embedding, cluster, filename):
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
        c = cluster,
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
    return (
        params.get('filename'),
        params.get('contact_pairs'),
        params.get('output'),
        int(params.get('dim_lat')),
        int(params.get('hidden_dim')),
        float(params.get('learning_rate')),
        int(params.get('epochs')),
    )


def generate_yaml_config(output_file):
    """Generates a YAML config file"""
    base_name, ext = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(output_file):
        output_file = f'{base_name}_{counter}{ext}'
        counter += 1

    config_data = {
        'filename': 'path/to/default/data/file',
        'contact_pairs': 'path/to/default/data/file',
        'output': 'path/to/output/files/',
        'dim_lat': 2,
        'hidden_dim': 80,
        'learning_rate': 1e-4,
        'epochs': 100,
    }
    comments = {
        'filename': 'File path containing the coordinates.',
        'contact_pairs': 'File path containing the contact pairs as indices.',
        'output': 'File path which will contain all output files.',
        'dim_lat': 'Dimensionality of the latent space',
        'hidden_dim': 'The dimension of the hidden layer',
        'learning_rate': 'Learning rate (usually 1e-2 - 1e-6)',
        'epochs': 'Number of epochs for the model to train.',
    }
    with open(output_file, 'w') as yaml_file:
        for key, value in config_data.items():
            if key in comments:
                yaml_file.write(f"# {comments[key]}\n")
            yaml.dump({key: value}, yaml_file, default_flow_style=False)
    return output_file


def plot_embedding(graphs, model, output, device, epochs, filename, all_loss):
    loader = DataLoader(graphs, batch_size=100, shuffle=True)
    # plot latent spac
    embeddings = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            embedding, _ = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
                )
            embeddings.append(embedding.cpu())
    embeddings_tensor = torch.cat(embeddings, dim=0)
    embeddings_np = embeddings_tensor.numpy()
    np.save(f"embeddings_ECC{output}.npy", embeddings_np)
    hist, xedges, yedges = np.histogram2d(
        embeddings_np[:, 0],
        embeddings_np[:, 1],
        density=True,
        bins=200,
    )
    hist = hist.T
    xedges = 0.5 * (xedges[1:] + xedges[:-1])
    yedges = 0.5 * (yedges[1:] + yedges[:-1])
    DeltaG = np.transpose(-np.log(hist))
    DeltaG -= np.nanmin(DeltaG)
    DeltaG[DeltaG == np.nanmax(DeltaG)] = None
    contour_min = int(np.floor(np.nanmin(DeltaG)))
    contour_max = int(np.ceil(np.nanmax(DeltaG)))
    contour_levels = np.linspace(contour_min, contour_max, 25)
    _, ax = plt.subplots(figsize=(2, 2))
    ax.contourf(
        xedges,
        yedges,
        DeltaG,
        levels=contour_levels,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    pplt.savefig(f"embedding_ECC_{output}.png")
    plt.close()
    crucial_dist = 38
    dist = np.loadtxt(filename)
    indx_closed = np.where(dist[:, crucial_dist] >= 1.3)[0]
    indx_open = np.where(dist[:, crucial_dist] <= 1.1)[0]
    index_intermediate = np.where(
        (1.1 < dist[:, crucial_dist]) & (dist[:, crucial_dist] < 1.3)
    )
    embedding_closed = embeddings_np[indx_closed]
    embedding_open = embeddings_np[indx_open]
    embedding_intermediate = embeddings_np[index_intermediate]
    _, ax = plt.subplots(figsize=(2, 2))
    ax.scatter(
        embedding_closed[:, 0],
        embedding_closed[:, 1],
        alpha=0.5,
        lw=0,
        s=1,
        label="closed",
    )
    ax.scatter(
        embedding_open[:, 0],
        embedding_open[:, 1],
        alpha=0.5,
        lw=0,
        s=1,
        label="open",
    )
    ax.scatter(
        embedding_intermediate[:, 0],
        embedding_intermediate[:, 1],
        alpha=0.5,
        lw=0,
        s=1,
        label="intermediate",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.legend()
    pplt.savefig(f"embedding_{output}_open_closed.png")
    plt.close()