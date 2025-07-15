```markdown
# Traffic Forecasting with GNNs

This repository contains scripts to preprocess traffic sensor data, train spatio‑temporal graph models, and visualize results on the METR‑LA and PEMS‑Bay benchmarks using PyTorch Geometric Temporal.

---

## Directory Structure

```

.
├── data/
│   ├── hitik\_metrla/           # Raw .pkl adjacency files for METR‑LA
│   ├── hitik\_pems/             # Raw .pkl adjacency files for PEMS‑Bay
│   ├── node\_values.npy         # Speed time‑series for METR‑LA
│   └── pems\_node\_values.npy    # Speed time‑series for PEMS‑Bay
├── models/                     # GNN & baseline model implementations
├── scripts/
│   ├── preprocess.py           # .pkl → PyG‑Temporal .pt preprocessing
│   ├── train.py                # Train & evaluate (incremental/cumulative)
│   ├── visualize.py            # Plot incremental-training bar charts
│   └── visualize\_ablation.py   # Plot GCN incr. vs. cumulative comparison
└── README.md                   # This usage guide

````

---

## Prerequisites

- Python 3.8+  
- PyTorch 2.x  
- `torch-geometric` & `torch-geometric-temporal`  
- `numpy`, `pandas`, `matplotlib`, `seaborn`  

Install with:

```bash
pip install torch torchvision torchaudio \
    torch-geometric torch-geometric-temporal \
    numpy pandas matplotlib seaborn
````

---

## 1. Data Preparation

1. Drop adjacency `.pkl` files into:

   * `data/hitik_metrla/`
   * `data/hitik_pems/`
2. Drop speed arrays into:

   * `data/node_values.npy`
   * `data/pems_node_values.npy`

---

## 2. Preprocessing

Convert each `.pkl` + `.npy` into a `StaticGraphTemporalSignal` `.pt`:

```bash
python scripts/preprocess.py \
  --data-dir data \
  --metrla-npy data/node_values.npy \
  --pems-npy  data/pems_node_values.npy \
  --horizon 1
```

This writes:

```
data/hitik_metrla/adj_mx_*.pt
data/hitik_pems/adj_mx_*.pt
```

---

## 3. Training & Evaluation

Train any model on METR‑LA or PEMS‑Bay:

```bash
python scripts/train.py \
  --seed 0 \
  --model gat \
  --dataset metrla \
  --train_size 60 \
  --val_size 20 \
  --window 99 \
  --resample 1 12 288 \
  --epochs 200 \
  --lr 0.01 \
  --hidden 64 \
  --heads 8 \
  --train_mode incremental \
  --data_dir data
```

* **`--model`**: `baseline` / `gcn` / `gat` / `cheb` / `appnp` / `sgc`
* **`--dataset`**: `metrla` or `pems`
* **`--train_size`** / **`--val_size`**: percentages for 60/20/20 split
* **`--window`**: 99 → 99+1 forecasting
* **`--resample`**: `1`=5 min, `12`=1 hour, `288`=1 day
* **`--train_mode`**: `incremental` or `cumulative`
* **`--data_dir`**: where the `.pt` files live

Outputs are saved as

```
results_<dataset>_<model>_<mode>_rmse.csv
```

---

## 4. Visualizing Incremental Results

Render adjacency‐variant comparisons:

```bash
python scripts/visualize.py
```

Generates:

* `incremental_metrla_adj_comparison.png`
* `incremental_pems_adj_comparison.png`

---

## 5. Visualizing GCN Ablation

Compare GCN incremental vs. cumulative training:

```bash
python scripts/visualize_ablation.py
```

Generates:

* `gcn_metrla_incr_vs_cumu.png`
* `gcn_pems_incr_vs_cumu.png`
