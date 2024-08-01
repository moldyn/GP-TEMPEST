# model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, Parameter, Sequential
from torch_geometric.nn import GATConv, NNConv, SAGPooling, Set2Set


class EdgeAttention(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention, self).__init__()
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.att_weight = Parameter(torch.Tensor(1, hidden_dim))
        self.edge_transform = nn.Linear(self.edge_feature_dim, self.hidden_dim)
        self.score_transform = nn.Linear(
            self.hidden_dim,
            self.edge_feature_dim,
        )

    def forward(self, edge_attr, edge_index, num_nodes):
        transformed_edge_attr = self.edge_transform(edge_attr)
        attention_scores = F.leaky_relu(
            self.score_transform(transformed_edge_attr),
        )
        attention_scores = torch.softmax(attention_scores, dim=0)  # normalize
        return edge_attr * attention_scores  # return weighted edges


class GraphEncoder(nn.Module):
    def __init__(self, layer_sizes):
        super(GraphEncoder, self).__init__()
        _, hidden_dim, out_features = layer_sizes
        edge_features = 1
        node_features = 4
        hidden_dim = int(hidden_dim)
        out_features = int(out_features)
        self.number_edges = None

        self.edge_attention = EdgeAttention(edge_features, hidden_dim)
        self.edge_mlp = Sequential(
            Linear(edge_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            Linear(hidden_dim, node_features * hidden_dim),
        )
        self.conv1 = NNConv(
            in_channels=int(node_features),
            out_channels=hidden_dim,
            nn=self.edge_mlp,
            aggr='mean',
        )
        self.pool1 = SAGPooling(hidden_dim)
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )
        self.pool2 = SAGPooling(hidden_dim)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.batch_norm2 = BatchNorm1d(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=5)
        self.fc = Linear(2 * hidden_dim, out_features)  # Set2Set output is 2 * hidden_dim
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr, batch):
        self.number_edges = edge_attr.size(0)
        edge_attr_attn = self.edge_attention(edge_attr, edge_index, x.size(0))
        edge_attr_attn = F.leaky_relu(edge_attr_attn)
        x = self.conv1(x, edge_index, edge_attr_attn)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x.relu_()
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x.relu_()
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.set2set(x, batch)
        x = self.fc(x)
        return x


class ConditionedGraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, max_edges):
        super(ConditionedGraphDecoder, self).__init__()
        input_dim = latent_dim + max_edges
        self.fc_expand = nn.Linear(input_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_embedding, mask):
        conditioned_input = torch.cat(
            (graph_embedding, mask.to(graph_embedding.device)),
            dim=1,
        )
        hidden_output = F.relu(self.fc_expand(conditioned_input))
        hidden_output = F.relu(hidden_output)
        full_output = self.fc_output(hidden_output)

        contacts_reconstructed = full_output[mask.bool()]
        return contacts_reconstructed


class GraphAutoencoder(nn.Module):
    def __init__(self, layer_sizes, max_edges):
        super(GraphAutoencoder, self).__init__()
        latent_dim = layer_sizes[-1]
        hidden_dim = layer_sizes[1]
        output_dim = max_edges

        self.encoder = GraphEncoder(layer_sizes)
        self.decoder = ConditionedGraphDecoder(
            latent_dim, hidden_dim, output_dim, max_edges,
        )

    def forward(self, x, edge_index, edge_attr, batch, one_hot_edges):
        graph_embedding = self.encoder(x, edge_index, edge_attr, batch)
        reconstructed_distances = self.decoder(graph_embedding, one_hot_edges)
        return graph_embedding, reconstructed_distances

