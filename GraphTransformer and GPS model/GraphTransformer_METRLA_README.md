
# Graph Transformer for Traffic Speed Prediction on METR-LA Dataset 📈

This repository contains a Python script that implements a **Graph Transformer model** to predict traffic speeds using the METR-LA dataset. The script explores how different adjacency matrix types and time strides affect the model's performance.

-----

### Table of Contents

  * [Overview]
  * [Features]
  * [Getting Started]
      * [Prerequisites])
      * [Data Preparation]
      * [Installation]
  * [Running the Script]
  * [Model Architecture]
  * [Experiment Configuration]
  * [Output and Results]

-----

## Overview

This script is an experimental framework for training and evaluating a Graph Transformer model on the METR-LA traffic speed dataset. It focuses on **predicting future traffic speeds** (12 time steps, or 60 minutes) based on past observations (12 time steps). A key goal is to test the model's performance across different graph structures (various adjacency matrices) and temporal sampling rates (strides).

The script handles everything from data loading and preprocessing to model definition, training, evaluation, and visualizing the predictions.

-----

## Features

  * **Graph Transformer Model**: Implements a custom Graph Transformer that integrates `GatedGraphConv` and `TransformerConv` layers for spatial and temporal processing.
  * **Positional Encoding**: Supports Random Walk Positional Encoding to capture the graph's structural information.
  * **METR-LA Dataset Integration**: Automatically extracts and loads the METR-LA dataset's node values and adjacency matrices.
  * **Flexible Adjacency Matrix Loading**: Includes a robust function to load adjacency matrices from various `.pkl` file formats, ensuring they are correctly shaped and symmetrized for METR-LA.
  * **Time Feature Engineering**: Adds sinusoidal time features to the input data to help the model recognize temporal patterns.
  * **Data Splitting and Sampling**: Divides the dataset into training (60%), validation (20%), and test (20%) sets. It also allows for configurable strides (1, 12, or 288 time steps) when creating data samples.
  * **Evaluation Metrics**: Computes Mean Absolute Error ($MAE$), Root Mean Squared Error ($RMSE$), and Mean Absolute Percentage Error ($MAPE$).
  * **Visualisation**: Generates plots to compare predicted vs. actual traffic speeds.
  * **Multi-GPU Support**: Automatically detects and uses multiple GPUs for faster training. 🚀
  * **Reproducibility**: Fixes random seeds to ensure consistent results across runs.

-----

## Getting Started

### Prerequisites

  * Python 3.x (developed with 3.11.0rc1)
  * `pip` (Python package installer)
  * A GPU with CUDA support is highly recommended for performance.

### Data Preparation

1.  Download the `METR-LA.zip` dataset.
2.  Place the zip file in the same directory as the script.
3.  The script will automatically extract the contents into a `METR-LA/` directory on its first run.

### Installation

1.  Clone this repository or download the script.
2.  Install the required Python packages using `pip`:
    ```bash
    pip install numpy torch torch_geometric scikit-learn matplotlib
    ```
    Ensure your `torch` and `torch_geometric` versions are compatible with your CUDA installation.

-----

## Running the Script

To execute the script, navigate to its directory in your terminal and run the following command.

```bash
python 100DAY_GraphTransformer_4ADJ_METRLA_adjusted_timesteps.ipynb
```

-----

## Model Architecture

The core model is the `GraphTransformer`, which processes the graph data.

1.  **Input Embedding**: Node features are embedded using a linear layer. If positional encoding (PE) is used, it's also transformed and concatenated with the node embeddings.
2.  **Graph Transformer Layers**: The model uses a stack of `GraphTransformerLayer` instances (default is 4 layers). Each layer contains:
      * A `GatedGraphConv` layer for spatial processing.
      * A `TransformerConv` layer for self-attention.
      * Two `LayerNorm` layers for normalization.
      * A feed-forward network with ReLU activation and dropout.
      * Residual connections are used throughout.
3.  **Output Layer**: The final output is produced by a Multi-Layer Perceptron (MLP) that predicts the future time steps.

The model's input channels are calculated as `in_timestep * 3` (1 speed feature + 2 time features), with hidden channels set to 64 and a PE dimension of 20.

-----

## Experiment Configuration

The script systematically evaluates the model under different configurations by iterating through:

  * **Adjacency Matrices**:
      * `adj_mx_knn_lane.pkl`
      * `adj_mx_knn.pkl`
      * `adj_mx_radius_lane.pkl`
      * `adj_mx_radius.pkl`
  * **Stride Options**: The stride determines the interval for generating data samples.
      * **Stride 1**: 5 minutes (default METR-LA interval).
      * **Stride 12**: 1 hour (12 \* 5 minutes).
      * **Stride 288**: 1 day (288 \* 5 minutes).

The model is trained for **20 epochs** for each combination of adjacency matrix and stride.

-----

## Output and Results

The script provides comprehensive output during and after execution.

### Console Output 💻

  * CUDA initialization and GPU usage details.
  * Information about the loaded adjacency matrix.
  * Data shapes and sample counts for training, validation, and test sets.
  * Training and validation loss for each epoch.
  * Final evaluation metrics ($MAE$, $RMSE$, $MAPE$) for the test set and a specific "100th Day" test set.
  * A final summary table of all results.

### Generated Plots 📊

The script creates directories named `plots_test_STRIDE_VALUE` and `plots_100th_STRIDE_VALUE`.

  * `prediction_test_...png`: Visualizes a sample prediction from the test set.
  * `prediction_100th_day_...png`: Visualizes a sample prediction from the 100th day data.
  * `full_100th_day_prediction_...png`: Shows a comprehensive plot of predicted vs. actual speeds for the entire 100th day, demonstrating the model's performance over a full 24-hour cycle.