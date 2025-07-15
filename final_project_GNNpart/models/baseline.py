import torch
import torch.nn as nn

class MovingAvgForecast(nn.Module):
    """
    Moving Average Baseline for traffic forecasting.
    Predicts each node's next value as the mean of its values over the input historical window.
    """
    def __init__(self):
        super(MovingAvgForecast, self).__init__()

    def fit(self, train_data_iter):
        """
        No fitting needed for moving average.
        This method exists for compatibility.
        """
        pass

    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Predict by averaging each node's historical window.

        Args:
            x (Tensor): Input features of shape [N, F] or [B, N, F].
                        Each row (or [b, n, :]) contains the history of a node.
        Returns:
            Tensor: Predictions of shape [N, 1] or [B, N]
        """
        if x.dim() == 2:
            # [N, F] → [N, 1]
            return x.mean(dim=1, keepdim=True)
        elif x.dim() == 3:
            # [B, N, F] → [B, N]
            return x.mean(dim=2)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")
