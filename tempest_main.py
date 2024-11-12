import os
import sys

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_gp
from network_gp import TEMPEST, MaternKernel


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
        out_file = utils_gp.generate_yaml_config('default_tempest_config.yaml')
        print(f'created a new config file {out_file}.')
        sys.exit()
    out_file = config.split('.')[0]

    (
        data_path, inducing_points_path, save_path, cuda, dim_input,
        dim_latent, layers_hidden, epochs, batch_size, learning_rate,
        weight_decay, beta, kernel_nu, kernel_scale,
    ) = utils_gp.yaml_config_reader(config)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    dataset = utils_gp.load_data(data_path)
    inducing_points = np.loadtxt(inducing_points_path)
    print(dataset)
    N_data_points = len(dataset[0])
    train_size = 0.9

    kernel = MaternKernel(nu=kernel_nu, scale=kernel_scale)
    tempest = TEMPEST(
        cuda=cuda,
        kernel=kernel,
        dim_input=dim_input,
        dim_latent=dim_latent,
        layers_hidden_encoder=layers_hidden,
        layers_hidden_decoder=layers_hidden[::-1],
        inducing_points=inducing_points,
        GP_w=beta,
        N_data=N_data_points,
    )
    tempest.train_model(
        dataset, train_size, learning_rate, weight_decay, batch_size, epochs,
    )




if __name__ == '__main__':
    main()