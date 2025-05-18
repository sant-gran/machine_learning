import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from hybrid_score import HybridSimilarityScorer

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.similarity = HybridSimilarityScorer(input_dim=out_channels * 2) # Concat size

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGEWithSimilarity(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.5, scorer_hidden_dim=32):
        super(GraphSAGEWithSimilarity, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels,  out_channels, dropout)
        self.similarity = HybridSimilarityScorer(input_dim=out_channels * 2, hidden_dim=scorer_hidden_dim)
    def forward(self, x, edge_index, edge_pairs=None):
        embeddings = self.encoder(x, edge_index)

        if edge_pairs is not None:
           # edge pairs should be shape [2, num_edges]
           node_i = embeddings[edge_pairs[0]]
           node_j = embeddings[edge_pairs[1]]
           sim_scores = self.similarity(node_i,node_j)
           return sim_scores # For link prediction or pairwise similarity
        else:
            return embeddings # For full embedding extraction (e.g., fusion)







