import os, sys
# Ensure project root is in Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import math
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

# Import model classes
from models.gcn import GCNForecast
from models.gat import GATForecast
from models.cheb import ChebNetForecast
from models.appnp import APPNPForecast
from models.sgc import SGCForecast
from models.baseline import MovingAvgForecast


def windowed_dataset(dataset, window_size: int, stride: int):
    """
    Build windowed features/targets sequences with given window_size and temporal stride.
    """
    orig_feats = dataset.features[::stride]
    orig_tgts  = dataset.targets[::stride]
    T = len(orig_feats)
    feat_list, tgt_list = [], []
    for i in range(T - window_size):
        x_seq = torch.stack(orig_feats[i: i + window_size], dim=1)  # [N, window]
        y_seq = orig_tgts[i + window_size]
        if y_seq.dim() == 1:
            y_seq = y_seq.unsqueeze(1)
        feat_list.append(x_seq)
        tgt_list.append(y_seq)
    return feat_list, tgt_list


def split_dataset(feats, tgts, train_size: float, val_size: float):
    """Temporal split into train/val/test by percentages."""
    n = len(feats)
    n_train = int(n * train_size / 100)
    n_val   = int(n * val_size   / 100)
    train = (feats[:n_train], tgts[:n_train])
    val   = (feats[n_train: n_train + n_val], tgts[n_train: n_train + n_val])
    test  = (feats[n_train + n_val:], tgts[n_train + n_val:])
    return train, val, test


def train_incremental(model, train_data, val_data, edge_index, edge_weight, device, epochs: int, lr: float):
    """Incremental training: update weights after each snapshot."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in zip(*train_data):
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            out = model(x, edge_index.to(device), edge_weight.to(device))
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_data[0])
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in zip(*val_data):
                x, y = x.to(device).float(), y.to(device).float()
                out = model(x, edge_index.to(device), edge_weight.to(device))
                val_loss += F.mse_loss(out, y).item()
        avg_val = val_loss / len(val_data[0])
        print(f"Epoch {epoch:02d} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_cumulative(model, train_data, val_data, edge_index, edge_weight, device, epochs: int, lr: float):
    """Cumulative training: aggregate loss and update weights once per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in zip(*train_data):
            x, y = x.to(device).float(), y.to(device).float()
            out = model(x, edge_index.to(device), edge_weight.to(device))
            epoch_loss += F.mse_loss(out, y)
        epoch_loss = epoch_loss / len(train_data[0])
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        avg_train = epoch_loss.item()
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in zip(*val_data):
                x, y = x.to(device).float(), y.to(device).float()
                out = model(x, edge_index.to(device), edge_weight.to(device))
                val_loss += F.mse_loss(out, y).item()
        avg_val = val_loss / len(val_data[0])
        print(f"Epoch {epoch:02d} | CumTrain MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def test_model(model, test_data, edge_index, edge_weight, device):
    model.eval()
    mse = 0.0
    with torch.no_grad():
        for x, y in zip(*test_data):
            x, y = x.to(device).float(), y.to(device).float()
            out = model(x, edge_index.to(device), edge_weight.to(device))
            mse += F.mse_loss(out, y).item()
    return mse / len(test_data[0])


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on with normalization and training mode ablation")
    parser.add_argument('--seed',       type=int, default=0, help='Random seed')
    parser.add_argument('--model',      type=str, choices=['baseline','gcn','gat','cheb','appnp','sgc'], required=True)
    parser.add_argument('--dataset',    type=str, choices=['metrla','pems'], required=True)
    parser.add_argument('--train_size', type=float, default=60)
    parser.add_argument('--val_size',   type=float, default=20)
    parser.add_argument('--window',     type=int, default=99)
    parser.add_argument('--resample',   nargs='+', type=int, default=[1,12,288])
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--hidden',     type=int, default=64)
    parser.add_argument('--heads',      type=int, default=8)
    parser.add_argument('--train_mode', type=str, choices=['incremental','cumulative'], default='incremental', help='Training mode')
    parser.add_argument('--data_dir',   type=str, default='data')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    pt_files = glob.glob(os.path.join(args.data_dir, f'hitik_{args.dataset}', '*.pt'))
    model_map = {
        'baseline': lambda: MovingAvgForecast(),
        'gcn':      lambda: GCNForecast(in_channels=args.window, hidden_channels=args.hidden, out_channels=1, dropout=0.0),
        'gat':      lambda: GATForecast(in_channels=args.window, hidden_channels=args.hidden, out_channels=1, heads=args.heads, dropout=0.0),
        'cheb':     lambda: ChebNetForecast(in_channels=args.window, hidden_channels=args.hidden, out_channels=1, K=2, dropout=0.0),
        'appnp':    lambda: APPNPForecast(in_channels=args.window, hidden_channels=args.hidden, out_channels=1, K=10, alpha=0.1, dropout=0.0),
        'sgc':      lambda: SGCForecast(in_channels=args.window, out_channels=1, K=2, dropout=0.0),
    }

    for pt_path in pt_files:
        name = os.path.basename(pt_path).replace('.pt', '')
        dataset = torch.load(pt_path, weights_only=False)
        edge_index, edge_weight = dataset.edge_index, dataset.edge_weight

        for stride in args.resample:
            feats, tgts = windowed_dataset(dataset, args.window, stride)
            train_data, val_data, test_data = split_dataset(feats, tgts, args.train_size, args.val_size)

            # normalization
            all_train_y = torch.cat([y for y in train_data[1]], dim=0)
            mean, std = all_train_y.mean(), all_train_y.std()
            train_data = ([(x-mean)/std for x in train_data[0]], [(y-mean)/std for y in train_data[1]])
            val_data   = ([(x-mean)/std for x in val_data[0]],   [(y-mean)/std for y in val_data[1]])
            test_data  = ([(x-mean)/std for x in test_data[0]],  [(y-mean)/std for y in test_data[1]])

            # init model
            model = model_map[args.model]().to(device)
            if args.model == 'baseline':
                model.fit(list(zip(train_data[0], train_data[1])))
            else:
                if args.train_mode == 'incremental':
                    model = train_incremental(model, train_data, val_data, edge_index, edge_weight, device, args.epochs, args.lr)
                else:
                    model = train_cumulative(model, train_data, val_data, edge_index, edge_weight, device, args.epochs, args.lr)

            test_mse_norm = test_model(model, test_data, edge_index, edge_weight, device)
            rmse_mph = math.sqrt(test_mse_norm) * std.item()
            print(f"{name} | stride={stride} | {args.model} | mode={args.train_mode} | RMSE: {rmse_mph:.4f} mph")
            results.append({'dataset': name, 'stride': stride, 'model': args.model, 'mode': args.train_mode, 'rmse_mph': rmse_mph})

    df = pd.DataFrame(results)
    out_csv = f"results_{args.dataset}_{args.model}_{args.train_mode}_rmse.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == '__main__':
    main()
