import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNForecast(torch.nn.Module):
    """
    Graph Convolutional Network for traffic forecasting,
    Two-layer GCN with ReLU activation and dropout.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super(GCNForecast, self).__init__()
        # First GCN layer: adds self-loops + symmetric normalization
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True)
        # Second GCN layer: output layer
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True, normalize=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of GCNForecast.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity [2, num_edges].
            edge_weight (Tensor, optional): Edge weights [num_edges].

        Returns:
            Tensor: Predictions for each node, shape [num_nodes, out_channels].
        """
        # Layer 1: GCN + ReLU + dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: final GCN
        x = self.conv2(x, edge_index, edge_weight)
        return x