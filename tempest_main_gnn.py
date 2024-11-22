import os
import sys

import click
import numpy as np
import torch

import tempest_utils
from tempest_gnn import TEMPEST_GNN, MaternKernel


@click.command(no_args_is_help=True)
@click.option(
    '--config',
    default=None,
    help='Path to the YAML configuration file.',
)
@click.option(
    '--generate_config',
    is_flag=True,
    help='Generate a default configuration file.',
)
def main(config, generate_config):
    """Run TEMPEST dimensionality reduction."""
    if generate_config:
        out_file = tempest_utils.generate_yaml_config_gnn(
            'default_tempest_config_gnn.yaml',
        )
        print(f'created a new config file {out_file}.')
        sys.exit()
    out_file = config.split('.')[0]

    (distances_path, distances_idxs, node_features, num_residues,
        inducing_points_path, save_path, cuda, dim_input,
        dim_latent, layers_hidden, epochs, batch_size, learning_rate,
        weight_decay, beta, kernel_nu, kernel_scale, header) = \
        tempest_utils.yaml_config_reader_gnn(config)
    print(header)

    basename_save = (
        f'neps_{epochs}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}'
        f'_b_{beta}_nu_{kernel_nu}_scale_{kernel_scale:.0f}'
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    dtype = torch.float32

    threshold = 1.05
    dataset, max_edges = tempest_utils.load_prepare_graphs(
        distances_path,
        distances_idxs,
        node_features,
        num_residues,
        threshold,
        dtype,
        torch.device('cuda' if cuda else 'cpu'),
    )
    inducing_points = np.loadtxt(inducing_points_path)
    N_data_points = len(dataset)
    train_size = 1

    kernel = MaternKernel(
        nu=kernel_nu,
        scale=kernel_scale,
        dtype=dtype,
    )
    tempest = TEMPEST_GNN(
        cuda=cuda,
        kernel=kernel,
        dim_input=dim_input,
        dim_latent=dim_latent,
        layers_hidden_encoder=layers_hidden,
        layers_hidden_decoder=layers_hidden[::-1],
        inducing_points=inducing_points,
        beta=beta,
        N_data=N_data_points,
        dtype=dtype,
        max_edges=max_edges,
    )
    model_path = f'{save_path}/model_{basename_save}.pt'
    if os.path.exists(model_path):
        tempest.load_state_dict(torch.load(model_path))
        print(f'Loaded already trained model from {model_path}')
    else:
        print(tempest)
        tempest.train_model(
            dataset, train_size, learning_rate, weight_decay, batch_size,
            epochs,
        )
        torch.save(tempest.state_dict(), model_path)

    embedding = tempest.extract_latent_space(dataset, batch_size)
    np.savetxt(
        f'{save_path}/embedding_{basename_save}.dat',
        embedding,
        header=header,
        comments='#',
        fmt='%.4f',
    )


if __name__ == '__main__':
    main()
