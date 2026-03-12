# -*- coding: utf-8 -*-
import click
import matplotlib.pyplot as plt
import msmhelper as mh
import numpy as np
import prettypyplot as pplt
from sklearn.cluster import KMeans

pplt.use_style(colors='summertimes')


def free_energy_landscape(data, nbins, nlevels):
    hist, xedges, yedges = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=nbins,
        density=True,
    )
    xedges = 0.5 * (xedges[1:] + xedges[:-1])
    yedges = 0.5 * (yedges[1:] + yedges[:-1])
    DeltaG = np.transpose(-np.log(hist))
    DeltaG -= np.nanmin(DeltaG)
    DeltaG[DeltaG == np.nanmax(DeltaG)] = None
    contour_min = int(np.floor(np.nanmin(DeltaG)))
    contour_max = int(np.ceil(np.nanmax(DeltaG)))
    contour_levels = np.linspace(contour_min, contour_max, nlevels)
    return DeltaG, xedges, yedges, contour_levels


def read_header_params(embedding_file):
    """Extract configuration parameters from embedding file header."""
    params = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            # Remove '# ' from the start of the line
            line = line.strip('# ')
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = value.strip()
    return params


@click.command(no_args_is_help='h')
@click.option(
    '--traj_original',
    type=click.Path(exists=True),
)
@click.option(
    '--traj_transformed',
    type=click.Path(exists=True),
)
@click.option(
    '--traj_embedding',
    type=click.Path(exists=True),
)
def create_plot(
    traj_original,
    traj_transformed,
    traj_embedding,
):
    original = np.loadtxt(traj_original)
    transformed = np.loadtxt(traj_transformed)
    latent = np.loadtxt(traj_embedding)

    kmeans = KMeans(n_clusters=4).fit(original)
    traj_ms = kmeans.labels_

    # rename the clusters by their y-position
    centroids = kmeans.cluster_centers_
    sorted_indices = np.argsort(-centroids[:, 1])
    sorted_traj_ms = np.zeros_like(traj_ms)
    for new_cluster, old_cluster in enumerate(sorted_indices):
        sorted_traj_ms[traj_ms == old_cluster] = new_cluster

    traj_ms_c = mh.md.corrections.dynamical_coring(
        sorted_traj_ms,
        lagtime=100,
        iterative=True,
    )
    traj_ms_c = np.array(traj_ms_c).flatten()
    unique_clusters = np.unique(traj_ms_c)
    color_map = {
        cluster: f'C{idx}' for idx, cluster in enumerate(unique_clusters)
    }

    deltag_l, xedges_l, yedges_l, contour_levels_l = free_energy_landscape(
        latent,
        100,
        15,
    )
    deltag_t, xedges_t, yedges_t, contour_levels_t = free_energy_landscape(
        transformed,
        100,
        15,
    )

    stride = 10

    fig, axs = plt.subplots(
        figsize=(3, 3),
        nrows=2,
        ncols=2,
        gridspec_kw={
            'wspace': 0.1,
            'hspace': 0.1,
        }
    )
    axs = axs.flatten()
    axs[0].scatter(
        original[:, 0][::stride],
        original[:, 1][::stride],
        s=1,
        c=[color_map[label] for label in traj_ms_c[::stride]],
        alpha=0.3,
        lw=0,
    )
    for num in unique_clusters:
        cluster_points = original[traj_ms_c == num]
        centroid = cluster_points.mean(axis=0)
        txt = axs[0].text(
            centroid[0],
            centroid[1],
            str(num),
            fontsize=16,
            ha='center',
            va='center',
        )
        pplt.add_contour(txt, 2, 'w')
    for num in unique_clusters:
        cluster_points = latent[sorted_traj_ms == num]
        centroid_ax1 = cluster_points.mean(axis=0)
        axs[1].scatter(
            latent[traj_ms_c == num][:, 0][::stride],
            latent[traj_ms_c == num][:, 1][::stride],
            s=1,
            c=color_map[num],
            alpha=0.3,
            lw=0,
        )
        txt = axs[1].text(
            centroid_ax1[0],
            centroid_ax1[1],
            str(num),
            fontsize=16,
            ha='center',
            va='center',
        )
        pplt.add_contour(txt, 2, 'w')
    axs[2].contourf(
        xedges_t,
        yedges_t,
        deltag_t,
        levels=contour_levels_t,
    )
    axs[2].contour(
        xedges_t,
        yedges_t,
        deltag_t,
        levels=contour_levels_t,
        cmap='gray',
    )
    axs[3].contourf(
        xedges_l,
        yedges_l,
        deltag_l,
        levels=contour_levels_l,
    )
    axs[3].contour(
        xedges_l,
        yedges_l,
        deltag_l,
        levels=contour_levels_l,
        cmap='gray',
    )

    # extract all important information from the header
    params = read_header_params(traj_embedding)
    beta = float(params['beta'])
    kernel_nu = float(params['kernel_nu'])
    kernel_scale = int(float(params['kernel_scale']))
    batch_size = int(params['batch_size'])
    learning_rate = float(params['learning_rate'])
    epochs = int(params['epochs'])
    neurons_ae = params['neurons_ae']
    ind_points = params['inducing_points_path']
    weight_decay = float(params['weight_decay'])
    ind_points = ind_points.split('/')[-1].split('.')[0]

    basename = (
        f'beta{beta}_nu{kernel_nu}_scale{kernel_scale}'
        f'_bs{batch_size}_lr{learning_rate}_wd{weight_decay}'
        f'_ep{epochs}'
        f'_neurons{"_".join(str(n) for n in eval(neurons_ae))}'
        f'_ind_points{ind_points}'
    )
    pplt.savefig(f'embedding_{basename}.png')


if __name__ == '__main__':
    create_plot()
