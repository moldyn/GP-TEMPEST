import os
import subprocess
import sys

import click
import numpy as np
import torch

import tempest_utils
from tempest_fc import TEMPEST, MaternKernel


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
        out_file = tempest_utils.generate_yaml_config('default_tempest_config.yaml')
        print(f'created a new config file {out_file}.')
        sys.exit()
    out_file = config.split('.')[0]

    (data_path, inducing_points_path, save_path, cuda, dim_input,
        dim_latent, layers_hidden, epochs, batch_size, learning_rate,
        weight_decay, beta, kernel_nu, kernel_scale, header) = \
        tempest_utils.yaml_config_reader(config)
    print(header)

    ind_points = inducing_points_path.split('/')[-1].split('.')[0]
    basename_save = (
        f'neps_{epochs}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}'
        f'_b_{beta}_nu_{kernel_nu}_scale_{kernel_scale:.0f}_'
        f'hidden_{"_".join(map(str, layers_hidden))}_ind_points_{ind_points}_'
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    dtype = torch.float64

    dataset, scale_factor = tempest_utils.load_prepare_data(data_path, dtype)
    inducing_points = np.loadtxt(inducing_points_path)
    N_data_points = len(dataset)
    train_size = 1
    adjusted_kernel_scale = kernel_scale / scale_factor
    adjusted_inducing_points = inducing_points / scale_factor

    kernel = MaternKernel(
        nu=kernel_nu,
        scale=adjusted_kernel_scale,
        dtype=dtype,
    )
    tempest = TEMPEST(
        cuda=cuda,
        kernel=kernel,
        dim_input=dim_input,
        dim_latent=dim_latent,
        layers_hidden_encoder=layers_hidden,
        layers_hidden_decoder=layers_hidden[::-1],
        inducing_points=adjusted_inducing_points,
        beta=beta,
        N_data=N_data_points,
        dtype=dtype,
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

    traj_original = "/data/evaluation/autoencoder/GraphAutoencoder/toy_model_GP/v4/traj_v4_original.dat"
    traj_transformed = "/data/evaluation/autoencoder/GraphAutoencoder/toy_model_GP/v4/traj_v4_transformed.dat"
    subprocess.run([
        "python3",
        "./test_run/plot_clusters_toymodel.py",
        "--traj_original", traj_original,
        "--traj_transformed", traj_transformed,
        "--traj_embedding", f'{save_path}/embedding_{basename_save}.dat',
    ])


if __name__ == '__main__':
    main()
