# Graphformer for Spatio-Temporal Traffic Forecasting

This repository implements a lightweight Graph Transformer (Graphformer) using PyTorch Geometric to forecast traffic speed data on the METR-LA and PEMS-BAY datasets. The model is trained using various custom adjacency matrices.

---

## 📁 Project Structure

```
Graphformer/
├── data/
│   ├── metr-la.h5
│   ├── pems-bay.h5
│   ├── adj_mx_knn.pkl
│   ├── adj_mx_radius.pkl
│   ├── adj_mx_knn_lane.pkl
│   ├── adj_mx_radius_lane.pkl
│   ├── adj_mx_knn_pems.pkl
│   ├── adj_mx_radius_pems.pkl
│   ├── adj_mx_knn_lane_pems.pkl
│   └── adj_mx_radius_lane_pems.pkl
├── graphformer_trainer.py      # Contains the model and training logic
├── run_graphformer.py          # Main script to execute experiments
└── graphformer_logs/           # Auto-created folder for result CSVs
```

---

## 📦 Requirements

Install all required Python packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy h5py
```

> ⚠️ For `torch-geometric`, follow their [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) specific to your CUDA & PyTorch version.

---

## 📂 Data Requirements

Place the following files inside the `data/` directory:

- **METR-LA Dataset:** `metr-la.h5`
- **PEMS-BAY Dataset:** `pems-bay.h5`
- **Adjacency Matrices:**  
  - For METR-LA:  
    `adj_mx_knn.pkl`, `adj_mx_radius.pkl`, `adj_mx_knn_lane.pkl`, `adj_mx_radius_lane.pkl`  
  - For PEMS-BAY:  
    `adj_mx_knn_pems.pkl`, `adj_mx_radius_pems.pkl`, `adj_mx_knn_lane_pems.pkl`, `adj_mx_radius_lane_pems.pkl`

These are required to match the filenames used in the scripts.

---

## 🚀 How to Run the Project

You should use the main script: `run_graphformer.py`.

```bash
python run_graphformer.py
```

This script will:

- Loop through all datasets (METR-LA and PEMS-BAY)
- Try four different adjacency matrices per dataset
- Evaluate on three time intervals: `5min`, `1h`, `1d`
- Train for a default of **20 epochs**
- Save training logs into `graphformer_logs/`

### Additional : Set Number of Epochs

You can optionally pass the number of training epochs:

```bash
python run_graphformer.py --epochs 50
```

---

## 📈 Output

Training logs are saved as CSV files in the `graphformer_logs/` folder:

```
graphformer_logs/
├── training_log_metrla_adj_mx_knn_5min.csv
├── training_log_pemsbay_adj_mx_radius_lane_1h.csv
└── ...
```

Each file contains:
- Epoch-wise MSE, MAE, RMSE, MAPE (both normalized and in mph)
- Helpful for post-analysis and visualization

---

## 🧠 Model Summary

The Graphformer model (`SimpleGraphFormer`) uses:
- Stacked `TransformerConv` layers from `torch_geometric`
- Input: 99 historical time steps (per sensor)
- Output: 1-step traffic speed prediction

---

## 📌 Notes

- Internally normalizes data using mean and std
- Resamples input data to `5min`, `1h`, or `1d` intervals
- Automatically handles train/val/test splits (60/20/20)
- Adjacency matrix is loaded from `.pkl` and converted via `dense_to_sparse`

---

## 🔬 Citation and Tools Used

- **Graphformer Paper**:  
  [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2207.06881)

---

