import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNetForecast(torch.nn.Module):
    """
    Chebyshev Spectral Graph Convolutional Network for traffic forecasting,
    Two-layer ChebNet with K-order polynomial filters, ReLU activation, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 2,
        dropout: float = 0.0
    ):
        super(ChebNetForecast, self).__init__()
        # First ChebConv layer: K-order Chebyshev polynomials
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        # Second ChebConv layer: output layer
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        # Layer 1: Chebyshev conv + ReLU + dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: Chebyshev conv
        x = self.conv2(x, edge_index, edge_weight)
        return x  # [num_nodes, out_channels]
