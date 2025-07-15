import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SGCForecast(torch.nn.Module):
    """
    Simplifying Graph Convolutional Network for traffic forecasting,
    Single-layer SGConv with K-order propagation and optional dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 2, dropout: float = 0.0):
        super(SGCForecast, self).__init__()
        # SGConv collapses weight matrices and non-linearities
        self.conv = SGConv(in_channels, out_channels, K=K, cached=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        # Optional input dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Single SGConv layer
        x = self.conv(x, edge_index, edge_weight)
        return x  # [num_nodes, out_channels]