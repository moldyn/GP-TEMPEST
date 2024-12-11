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
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

pplt.use_style()


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


def _gaussian_decay(d, a=1.2):
    return np.exp(- (d / a)**2)


def load_prepare_graphs(
    file_contact_distances,
    file_idx_contact_pairs,
    file_node_features,
    num_residues,
    threshold_distance,
    dtype,
    device,
):
    """Load and prepare protein structure graphs."""
    all_contact_distances = np.loadtxt(file_contact_distances)
    times = torch.tensor(
        np.arange(len(all_contact_distances)), dtype=dtype,
    ).reshape(-1, 1)
    max_edges = np.shape(all_contact_distances)[1]
    contact_pairs = np.loadtxt(file_idx_contact_pairs, dtype=int)
    node_features = np.loadtxt(file_node_features).reshape(-1, num_residues, 4)
    node_features = torch.tensor(node_features, dtype=dtype)

    graphs = []
    for num, contact_distances in enumerate(all_contact_distances):
        contact_idxs = np.where(contact_distances < threshold_distance)[0]
        valid_contact_pairs = contact_pairs[contact_idxs]
        valid_contact_distances = contact_distances[contact_idxs]
        mask = torch.zeros(max_edges, dtype=dtype)
        mask[contact_idxs] = 1
        edge_index = torch.tensor(
            valid_contact_pairs, dtype=torch.long,
        ).t().contiguous()
        edge_attr = torch.tensor(
            _gaussian_decay(valid_contact_distances), dtype=dtype,
        ).view(-1, 1)
        # print(f"Graph {num}:")
        # print(f"  node_features shape: {node_features[num].shape}")
        # print(f"  edge_index shape: {edge_index.shape}")
        # print(f"  edge_attr shape: {edge_attr.shape}")
        # print(f"  mask shape: {mask.shape}")
        # print(f"  times shape: {times[num].shape}")
        data = Data(
            x=node_features[num],
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask=mask,
            t=times[num]
        )
        graphs.append(data.to(device))
    return graphs, max_edges


# def load_prepare_graphs(
#     file_contact_distances,
#     file_idx_contact_pairs,
#     file_node_features,
#     num_residues,
#     threshold_distance,
#     dtype,
# ):
#     """Load and prepare protein structure graphs.
#
#     Creates PyG Data objects from protein structure information including
#     contact distances, residue pairs, and node features (dihedrals).
#     """
#     all_contact_distances = np.loadtxt(file_contact_distances)
#     times = torch.tensor(
#         np.arange(len(all_contact_distances)), dtype=dtype,
#     ).reshape(-1, 1)
#     max_edges = np.shape(all_contact_distances)[1]
#     contact_pairs = np.loadtxt(file_idx_contact_pairs, dtype=int)
#     node_features = np.loadtxt(file_node_features).reshape(-1, num_residues, 4)
#     node_features = torch.tensor(node_features, dtype=dtype)
#     graphs = []
#     one_hot_vector_list = []
#     for num, contact_distances in enumerate(all_contact_distances):
#         contact_idxs = np.where(contact_distances < threshold_distance)[0]
#         valid_contact_pairs = contact_pairs[contact_idxs]
#         valid_contact_distances = contact_distances[contact_idxs]
#         one_hot_vector = np.zeros(max_edges)
#         one_hot_vector[contact_idxs] = 1
#         one_hot_vector_list.append(one_hot_vector)
#         edge_index = torch.tensor(
#             valid_contact_pairs, dtype=torch.long,
#         ).t().contiguous()
#         edge_attr = torch.tensor(
#             _gaussian_decay(valid_contact_distances), dtype=dtype,
#         ).view(-1, 1)
#         graph = Data(
#             x=node_features[num],
#             edge_index=edge_index,
#             edge_attr=edge_attr
#         )
#         graphs.append(graph)
#     return graphs, one_hot_vector_list, times


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


def yaml_config_reader_gnn(config: str):
    """Parse all parameters using the yaml file."""
    with open(config, 'r') as stream:
        params = yaml.safe_load(stream)

    # Create header string with all parameters for reproducibility
    header = (
        "# TEMPEST model configuration Parameters:\n"
        f"# distances_path: {params.get('distances_path')}\n"
        f"# distances_idxs: {params.get('distances_idxs')}\n"
        f"# node_features: {params.get('node_features')}\n"
        f"# num_residues: {params.get('num_residues')}\n"
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
        params.get('distances_path'),
        params.get('distances_idxs'),
        params.get('node_features'),
        params.get('num_residues'),
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


def generate_yaml_config_gnn(output_file):
    """Generates a YAML config file"""
    base_name, ext = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(output_file):
        output_file = f'{base_name}_{counter}{ext}'
        counter += 1

    config_data = {
        'distances_path': 'path/to/default/data/file',
        'distances_idxs': 'path/to/default/index/file',
        'node_features': 'path/to/default/node_features/file',
        'num_residues': 35,
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
        'distances_path': 'File path containing the distances.',
        'distances_idxs': 'File path containing the indices of the distances.',
        'node_features': 'File path containing the node features.',
        'num_residues': 'Number of residues in the protein.',
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