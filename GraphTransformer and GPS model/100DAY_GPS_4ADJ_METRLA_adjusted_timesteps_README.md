The provided script, 100DAY_GPS_4ADJ_METRLA_adjusted_timesteps.ipynb, is designed to train and evaluate a GPS (General, Powerful, Scalable) Graph Transformer model for node-level traffic speed regression on the METR-LA dataset. It systematically tests the model's performance across various adjacency matrix types and temporal granularities.
Here's a breakdown of the script's key functionalities:
• Initial Setup and Dependencies
    ◦ The script imports necessary libraries such as numpy, torch, torch.nn, torch.optim, torch_geometric.nn, torch_geometric.data, torch_geometric.loader, torch_geometric.transforms, sklearn.preprocessing, matplotlib.pyplot, pickle, os, zipfile, and math.
    ◦ It checks for CUDA availability and initialises it if found, indicating the number of available GPUs. It also prints the torch_geometric version.
    ◦ Random seeds are fixed for reproducibility using torch.manual_seed(42) and np.random.seed(42).
    ◦ The script attempts to extract the METR-LA.zip dataset into a METR-LA directory if it doesn't already exist.
• Data Loading and Preprocessing
    ◦ load_adjacency_matrix(adj_matrix_path): This function is crucial for loading various adjacency matrices. It handles different pickle file formats (tuple, dictionary, list) to extract a valid square 2D matrix. If loading a custom matrix fails, it falls back to a default adj_mat.npy file.
    ◦ load_metr_la_data(data_dir, num_days, custom_adj_path): This function loads node values for a specified number of days (defaulting to 99 days for training). It extracts traffic speed data and scales it using MinMaxScaler. It then converts the adjacency matrix into edge_index (indicating connections) and edge_weight tensors, moving them to the specified device (GPU or CPU).
    ◦ add_time_features(features, current_time_step_index, ...): This function adds cyclical time-of-day features (sine and cosine components) to the input features, calculated based on the global_time_step. This enriches the data with temporal context.
    ◦ create_data_samples(node_values, in_timestep, out_timestep, stride): This function generates input-output samples using a sliding window approach. It permutes and reshapes the features to include the added time features, resulting in a final feature tensor with shape (num_samples, num_nodes, -1).
    ◦ create_100th_day_data(...): This dedicated function prepares data specifically for the 100th day of the METR-LA dataset, which is used for separate evaluation. It loads this specific day's data, scales it, and creates samples, carefully adjusting global time indices for correct time feature generation.
    ◦ Data Splitting: The loaded data (features and labels) is chronologically split into 60% training, 20% validation, and 20% testing sets to maintain temporal causality.
    ◦ Positional Encodings: AddRandomWalkPE(walk_length=20, attr_name='pe') is applied as a PyTorch Geometric transform to add Random Walk Positional Encodings to the graph data.
    ◦ DataLoader: PyTorch Geometric DataLoader instances are created for training, validation, testing, and the 100th-day data, with a batch_size of 32.
• GPS Model Architecture (GPSGraphTransformer)
    ◦ GPSLayer: This class defines a single layer of the GPS model. It incorporates:
        ▪ GatedGraphConv: For local message passing.
        ▪ TransformerConv: For global attention.
        ▪ LayerNorm and nn.Sequential (FFN): For normalization and a feed-forward network, along with dropout and residual connections.
    ◦ GPSGraphTransformer: This is the main model that stacks multiple GPSLayer instances (default 4 layers). It includes an initial node_emb (embedding layer) for input features, a pe_lin for positional encodings if use_pe is true, and an output mlp (Multi-Layer Perceptron) to produce the final predictions. The forward method processes inputs x, edge_index, edge_weight, pe (positional encodings), and batch. The output is reshaped to (batch_size, num_nodes, out_channels).
    ◦ The model is initialised with in_channels=in_timestep * 3 (for speed, sin/cos time, and node degree features) and out_channels=out_timestep (which is 12).
• Training and Evaluation Loop
    ◦ The model uses the AdamW optimizer with a learning rate of 0.001 and weight decay of 0.01.
    ◦ nn.MSELoss() is used as the training criterion.
    ◦ The script runs for 20 epochs.
    ◦ During training, it iterates through the train_loader, performs forward and backward passes, and updates model parameters. It also calculates and prints training and validation loss per epoch.
    ◦ compute_metrics(pred, target, scaler): This function calculates MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error) on the unscaled data. It handles potential division by zero for MAPE by adding a small epsilon and masking zero targets.
    ◦ visualize_predictions(...): This function plots predicted vs. actual traffic speeds for the first few sensors (num_sensors=5) in the first sample of a batch.
    ◦ visualize_full_100th_day(...): This function reconstructs and visualises the full 100th day's predictions by concatenating and averaging overlapping prediction windows. It then unscales the data and plots the results for selected sensors over 24 hours.
    ◦ The evaluate function calculates and prints the metrics for both the test set and the 100th-day set. It also calls visualize_predictions for the first batch of the test set.
    ◦ After evaluation, visualize_full_100th_day is called to save the full day's prediction plots.
• Execution Flow and Results Storage
    ◦ The script iterates through a predefined list of four adjacency matrices: adj_mx_knn_lane.pkl, adj_mx_knn.pkl, adj_mx_radius_lane.pkl, and adj_mx_radius.pkl.
    ◦ For each adjacency matrix, it iterates through three stride options: 5 minutes (stride 1), 1 hour (stride 12), and 1 day (stride 288).
    ◦ Results for each configuration (adjacency file and stride) are stored in a results dictionary, containing test and 100th-day metrics (MAE, RMSE, MAPE).
    ◦ Finally, a summary of all results is printed to the console