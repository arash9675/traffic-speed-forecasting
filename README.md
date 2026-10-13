# Traffic Speed Forecasting with GNNs and Graph Transformers

Lab project for *Data Science in Practice* (University of Bonn, SS 2025) by Hitik Panchal, Haofan Wen and Arash Astanboos.

We forecast traffic speed on the **METR-LA** and **PEMS-Bay** sensor networks with classic GNNs (GCN, ChebNet, GAT, APPNP, SGC), **Graphormer**, **Graph Transformer** and **GPS**, at three temporal resolutions (5 min, 1 hour, 1 day). We compare four custom adjacency matrices: `KNN`, `Radius`, `KNN+Lane` and `Radius+Lane`. Everything is implemented in PyTorch and PyTorch Geometric and evaluated with RMSE (mph).

📄 **Full report:** [report/Spatio_Temporal_Lab_Report.pdf](report/Spatio_Temporal_Lab_Report.pdf)

## Key results

- All learned models beat the historical average baseline, by the widest margin at short horizons.
- **ChebNet** is the most stable GNN across resolutions, followed by APPNP. SGC is consistently the weakest.
- **Graphormer** reaches about 1.7–2.0 mph RMSE at 5 min on PEMS-Bay, against about 9.5 mph for the baseline.
- **GPS vs Graph Transformer:** GPS is slightly better on METR-LA at 5 min and 1 day. On PEMS-Bay the two are nearly identical (best is 4.65 mph at 1 h with `KNN`, and 3.77 mph at 1 day with `KNN+Lane`).
- Lane-aware adjacencies help most for GCN and ChebNet and at fine time scales. On PEMS-Bay the choice of adjacency matters little.

## Repository layout

| Folder | Contents |
|---|---|
| `Data Generation/` | Builds the custom adjacency matrices (sensor locations, lane counts, QGIS maps) |
| `final_project_GNNpart/` | GNN models (GCN, ChebNet, GAT, APPNP, SGC), training scripts and RMSE results |
| `Graphformer_Model/` | Graphformer implementation and trainer |
| `GraphTransformer and GPS model/` | Final Graph Transformer and GPS notebooks for both datasets, including the PEMS-Bay ablation |
| `experiments/` | Earlier iterations and exploratory notebooks (first GPS/GT prototypes, 100-day runs, plotting) |
| `report/` | Final report (PDF and LaTeX source), final presentation, progress reports and figures |

Each model folder has its own README covering setup, dependencies and how to run it.

> **Note:** `report/latex/main.tex` references some figures that live in other folders or weren't kept (e.g. `MET Full.png`, `Project/*.png`). Use the PDF as the source of truth.
