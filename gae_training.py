# gae_training.py
import os
import random
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils
from model import GraphAutoencoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import prettypyplot as pplt
    pplt.use_style()
except ImportError:
    print("prettypyplot not found, using default matplotlib style")

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"~~~   Runs on {device}   ~~~")
if str(device) == "cuda":
    print('empty cache')
    torch.cuda.empty_cache()


class GraphDataset(Dataset):
    def __init__(self, graphs, masks):
        self.graphs = graphs
        self.masks = torch.tensor(masks, dtype=torch.float)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.masks[idx]


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = []
    embeddings_training = []
    noise_std = random.uniform(0.05, 0.15)

    for batched_data, masks in loader:
        batched_data = batched_data.to(device)
        noise = torch.randn(batched_data.edge_attr.size()) * noise_std
        noisy_edge_attrs = batched_data.edge_attr + noise.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        with autocast():
            embedding, reconstructed_distances = model(
                batched_data.x,
                batched_data.edge_index,
                noisy_edge_attrs,
                batched_data.batch,
                masks,
            )
            loss = criterion(
                reconstructed_distances.view(-1, 1),
                batched_data.edge_attr.view(-1, 1),
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss.append(loss.item())
        embeddings_training.append(embedding.detach())

    embeddings_training = torch.cat(embeddings_training, dim=0)
    embeddings_training = embeddings_training.detach().cpu().numpy()
    return np.mean(total_loss), embeddings_training


def train(
    dataset,
    hidden_dim,
    dim_lat,
    max_edges,
    learning_rate,
    epochs,
    output,
):
    layer_sizes = np.array([4, int(hidden_dim), dim_lat])
    model = GraphAutoencoder(layer_sizes, max_edges).to(device)
    scaler = GradScaler()

    loader = DataLoader(
        dataset,
        batch_size=80,
        shuffle=True,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=10,
        verbose=True,
    )
    criterion = nn.MSELoss(reduction='mean')

    all_loss = []
    epochs_range = range(epochs)
    for epoch in tqdm(epochs_range):
        loss, embeddings_training = train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            scaler,
        )
        all_loss.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss*1000:.4f}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(device) / (1024 ** 2)} MB")
        scheduler.step(loss)

        torch.cuda.empty_cache()

        if epoch % 1 == 0:
            embeddings_training = []
            for batched_data, masks in loader:
                optimizer.zero_grad()
                batched_data = batched_data.to(device)
                masks.to(device)
                with torch.no_grad(), autocast():
                    embedding, _ = model(
                        batched_data.x,
                        batched_data.edge_index,
                        batched_data.edge_attr,
                        batched_data.batch,
                        masks,
                    )
                embeddings_training.append(embedding)
            del embedding
            del _
            embeddings_training = torch.cat(embeddings_training, dim=0)
            embeddings_training = embeddings_training.detach().cpu().numpy()
            np.savetxt(
                f'latent_space_invdistances/embedding_{epoch}.dat',
                embeddings_training,
                fmt='%.3f',
            )

            _, ax = plt.subplots(figsize=(2, 2))
            ax.scatter(
                embeddings_training[:, 0],
                embeddings_training[:, 1],
                alpha=0.5,
                lw=0,
                s=1,
            )
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            pplt.savefig(f'{output}_embedding_epoch{epoch}.png')

    np.savetxt(f"loss_{output}.txt", all_loss)
    _, ax = plt.subplots()
    ax.plot(all_loss[1:], label="train")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    pplt.legend()
    pplt.savefig(f"loss_{output}.pdf")
    return model, all_loss


@click.command(
    no_args_is_help='-h',
    help='''Graph based autoencoder.''',
)
@click.option(
    '--config',
    default=None,
    help='Path to the YAML configuration file',
)
@click.option(
    '--generate_config',
    is_flag=True,
    help='Generates a YAML config file with some default values.',
)
@click.option(
    '--distribution',
    required=False,
    type=click.STRING,
    help='Plot the distribution of the distances and then quit. Output name.'
)
def main(config, generate_config, distribution):
    if generate_config:
        utils.generate_yaml_config('gae_config.yaml')
        sys.exit()
    (
        filename,
        contact_pairs,
        output,
        dim_lat,
        hidden_dim,
        learning_rate,
        epochs,
    ) = utils.yaml_config_reader(config)
    if distribution:
        utils.plot_distribution(filename, distribution)
        sys.exit()

    graphs, masks = utils.load_data(
        filename,
        contact_pairs,
        '/data/evaluation/autoencoder/GraphAutoencoder/data_T4L/node_features_100ps.txt',
    )
    dataset = GraphDataset(graphs, masks)
    max_edges = np.shape(masks)[1]

    model, all_loss = train(
        dataset,
        hidden_dim,
        dim_lat,
        max_edges,
        learning_rate,
        epochs,
        output,
    )
    torch.save(
        model.state_dict(),
        f'./latent_space_invdistances/graph_autoencoder_ECC_parameters_{output}.pth',
    )


if __name__ == "__main__":
    main()
