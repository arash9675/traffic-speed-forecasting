import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATForecast(torch.nn.Module):
    """
    Graph Attention Network for traffic forecasting,
    Two-layer multi-head GAT with ELU activation and configurable dropout.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0
    ):
        super(GATForecast, self).__init__()
        # First GAT layer: multi-head, concat output
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True
        )
        # Second GAT layer: single head, no concat to match out_channels
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            add_self_loops=True,
            concat=False
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        # Input dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 1: attention + ELU
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Dropout between layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: attention
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_channels]
