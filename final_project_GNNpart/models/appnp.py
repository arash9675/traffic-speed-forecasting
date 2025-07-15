import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class APPNPForecast(nn.Module):
    """
    Approximate Personalized Propagation of Neural Predictions for traffic forecasting,
    MLP encoder + APPNP propagation.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.0
    ):
        super(APPNPForecast, self).__init__()
        # MLP encoder: two linear layers
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        # APPNP propagation layer
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        # MLP encoding
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # Propagation
        x = self.prop(x, edge_index, edge_weight)
        return x  # [num_nodes, out_channels]