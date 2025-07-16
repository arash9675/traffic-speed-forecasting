import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import h5py
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import TransformerConv
from torch.utils.data import DataLoader, TensorDataset
import os

from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path("graphformer_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleGraphFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels))
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.out(x).squeeze(-1)

def laplacian_pe(adj_mx, k=10):
    N = adj_mx.shape[0]
    deg = adj_mx.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    L = np.eye(N) - D_inv_sqrt @ adj_mx @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L)
    return torch.tensor(eigvecs[:, 1:1+k], dtype=torch.float32)

def create_sequences(data, T_in=99, T_out=1):
    X, Y = [], []
    for t in range(len(data) - T_in - T_out):
        x = data[t:t+T_in]
        y = data[t+T_in + T_out - 1]
        X.append(x.T)
        Y.append(y)
    return np.stack(X), np.stack(Y)

def load_and_preprocess_graphformer(h5_path, pkl_path, time_freq='5min', hist_steps=99, pred_steps=1):
    df = pd.read_hdf(h5_path)
    data = df.values
    data = np.nan_to_num(data)

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    data_norm = (data - mean) / (std + 1e-5)
    data_norm = np.nan_to_num(data_norm)

    with open(pkl_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)

    df_speed = pd.DataFrame(data_norm, index=pd.date_range(start='2012-01-01', periods=len(data), freq='5min'))
    df_agg = df_speed.resample(time_freq).mean()
    df_agg = df_agg.fillna(0)

    data_resampled = df_agg.values
    X_np, Y_np = create_sequences(data_resampled, T_in=hist_steps, T_out=pred_steps)

    num_samples = X_np.shape[0]
    train_len = int(num_samples * 0.6)
    val_len = int(num_samples * 0.2)
    test_len = num_samples - train_len - val_len

    X_train, Y_train = X_np[:train_len], Y_np[:train_len]
    X_val, Y_val = X_np[train_len:train_len + val_len], Y_np[train_len:train_len + val_len]
    X_test, Y_test = X_np[train_len + val_len:], Y_np[train_len + val_len:]

    return (torch.tensor(X_train).float(), torch.tensor(Y_train).float(),
            torch.tensor(X_val).float(), torch.tensor(Y_val).float(),
            torch.tensor(X_test).float(), torch.tensor(Y_test).float(),
            edge_index, edge_weight, mean, std)
    
def evaluate_metrics(y_pred, y_true, std, mean):
    std_expanded = std.repeat(y_pred.shape[0] // std.shape[0])
    mean_expanded = mean.repeat(y_pred.shape[0] // mean.shape[0])
    
    y_pred_denorm = y_pred * std_expanded + mean_expanded
    y_true_denorm = y_true * std_expanded + mean_expanded

    mask = y_true_denorm.abs() > 1e-2  # mask out near-zero ground truth

    mse_norm = torch.mean((y_pred - y_true)**2).item()
    rmse_norm = torch.sqrt(torch.mean((y_pred - y_true)**2)).item()
    mse_mph = torch.mean((y_pred_denorm - y_true_denorm)**2).item()
    rmse_mph = torch.sqrt(torch.mean((y_pred_denorm - y_true_denorm)**2)).item()
    mae_mph = torch.mean(torch.abs(y_pred_denorm - y_true_denorm)).item()
    mape = torch.mean(
        torch.abs((y_pred_denorm[mask] - y_true_denorm[mask]) / y_true_denorm[mask])
    ) * 100

    return {
        "mse_norm": mse_norm,
        "rmse_norm": rmse_norm,
        "mae_mph": mae_mph,
        "mape%": mape.item(),
        "mse_mph": mse_mph,
        "rmse_mph": rmse_mph
    }

# def evaluate_baseline(X, Y, method, std, mean):
#     if method == 'last':
#         y_pred = X[:, :, -1]
#     elif method == 'mean':
#         y_pred = X[:, :, :-2].mean(dim=2)
#     else:
#         raise ValueError("Choose 'last' or 'mean'")

#     y_pred_flat = y_pred.reshape(-1)
#     y_true_flat = Y.reshape(-1)

#     return evaluate_metrics(
#         y_pred_flat.to(device),
#         y_true_flat.to(device),
#         torch.tensor(std).to(device),
#         torch.tensor(mean).to(device)
#     )

def train_model(X_train, Y_train, X_val, Y_val, edge_index, mean, std, batch_size=64, epochs=2, lr=0.001):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    model = SimpleGraphFormer(in_channels=X_train.shape[2], hidden_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    metrics_log = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_flat = xb.reshape(-1, xb.shape[-1])
            pred = model(xb_flat, edge_index.to(device)).reshape(xb.shape[0], -1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                xb_flat = xb.reshape(-1, xb.shape[-1])
                pred = model(xb_flat, edge_index.to(device)).reshape(xb.shape[0], -1)
                metrics = evaluate_metrics(pred.reshape(-1), yb.reshape(-1), torch.tensor(std).to(device), torch.tensor(mean).to(device))
        metrics["epoch"] = epoch + 1
        metrics_log.append(metrics)
        print(f"[{epoch+1}] MSE: {metrics['mse_norm']:.4f} | MAE: {metrics['mae_mph']:.2f} | MAPE: {metrics['mape%']:.2f}%")

    return model, metrics_log

def save_training_log_to_csv(loss_log, save_path):
    df_log = pd.DataFrame(loss_log)
    df_log.to_csv(LOG_DIR / save_path, index=False)
    print(f"Saved to {LOG_DIR / save_path}")
    
# def save_baseline_log_to_csv(baseline_log, save_path):
#     df = pd.DataFrame(baseline_log)
#     df.to_csv(save_path, index=False)
#     print(f"Baseline metrics saved to {save_path}")

def main(epoch_no=20):
    print(f"\n=== Number of Epochs = {epoch_no} ===")
    
    configs = [
        ("metr-la.h5", ["adj_mx_knn_lane.pkl", "adj_mx_radius_lane.pkl", "adj_mx_knn.pkl", "adj_mx_radius.pkl"], "metrla"),
        ("pems-bay.h5", ["adj_mx_knn_lane_pems.pkl", "adj_mx_radius_lane_pems.pkl", "adj_mx_knn_pems.pkl", "adj_mx_radius_pems.pkl"], "pemsbay")
    ]

    intervals = ['5min', '1h', '1d']

    for h5_file, adj_list, dataset_name in configs:
        for adj_file in adj_list:
            for interval in intervals:
                # print(f"\n=== Baselines for {interval} ({adj_file}) ===")
                h5_path = os.path.join("data", h5_file)
                pkl_path = os.path.join("data", adj_file)
                X_train, Y_train, X_val, Y_val, X_test, Y_test, edge_index, edge_weight, mean, std = load_and_preprocess_graphformer(h5_path, pkl_path, time_freq=interval)

                # baseline_log = []
                # for method in ['last', 'mean']:
                #     metrics = evaluate_baseline(X, Y, method, mean, std)
                #     print(f"{method.upper()} → MAE: {metrics['mae_mph']:.2f} | MAPE: {metrics['mape%']:.2f}% | RMSE: {metrics['rmse_mph']:.2f}")
                #     metrics['method'] = method
                #     metrics['adj_file'] = adj_file
                #     metrics['interval'] = interval
                #     baseline_log.append(metrics)

                # # Save baseline CSV
                # baseline_log_filename = f"baseline_log_{dataset_name}_{adj_file.replace('.pkl', '')}_{interval}.csv"
                # save_baseline_log_to_csv(baseline_log, baseline_log_filename)

                print(f"\n=== Training for {interval} for {dataset_name} and adj_mx {adj_file} ===")
                model, log = train_model(X_train, Y_train, X_val, Y_val, edge_index, mean, std, epochs=epoch_no)
                log_filename = f"training_log_{dataset_name}_{adj_file.replace('.pkl', '')}_{interval}.csv"
                save_training_log_to_csv(log, log_filename)

if __name__ == '__main__':
    main()