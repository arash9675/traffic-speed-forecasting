import os
import argparse
import pickle
import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def process_pkl(pkl_path: str, npy_path: str, horizon: int):
    # Load adjacency and sensor mapping
    with open(pkl_path, 'rb') as f:
        sensor_ids, id2idx, adj = pickle.load(f)
    # Symmetrize and binarize adjacency
    adj = adj + adj.T
    adj[adj > 0] = 1
    # Build edge_index and edge_weight
    row, col = np.nonzero(adj)
    edge_index = torch.from_numpy(np.vstack((row, col))).long()     # [2, E]
    edge_weight = torch.from_numpy(adj[row, col].astype(np.float32))  # [E]
    # Load node features
    nv = np.load(npy_path)  # shape [T, N, C] or [T, N]
    if nv.ndim == 3 and nv.shape[2] >= 1:
        nv = nv[:, :, 0]
    T, N = nv.shape
    # Build feature and target lists
    features_list, targets_list = [], []
    for t in range(T - horizon):
        features_list.append(torch.tensor(nv[t], dtype=torch.float))
        targets_list.append(torch.tensor(nv[t + horizon], dtype=torch.float))
    # Create PGT dataset and save
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features_list,
        targets=targets_list,
    )
    save_path = os.path.splitext(pkl_path)[0] + '.pt'
    torch.save(dataset, save_path)
    return len(features_list), N, save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch preprocess .pkl to .pt")
    parser.add_argument('--data-dir', type=str, default='data',
                        help="Root data directory containing hitik_metrla and hitik_pems")
    parser.add_argument('--metrla-npy', type=str, default='data/node_values.npy',
                        help="Path to METR-LA node_values.npy")
    parser.add_argument('--pems-npy', type=str, default='data/pems_node_values.npy',
                        help="Path to PEMS-Bay node_values.npy")
    parser.add_argument('--horizon', type=int, default=1,
                        help="Prediction horizon (number of steps ahead)")
    args = parser.parse_args()

    for sub in ['hitik_metrla', 'hitik_pems']:
        folder = os.path.join(args.data_dir, sub)
        npy_path = args.metrla_npy if sub == 'hitik_metrla' else args.pems_npy
        for fname in os.listdir(folder):
            if not fname.endswith('.pkl'):
                continue
            pkl_path = os.path.join(folder, fname)
            snaps, nodes, out_path = process_pkl(pkl_path, npy_path, args.horizon)
            print(f"Processed {fname} -> {os.path.basename(out_path)}: {snaps} snapshots, {nodes} nodes.")

