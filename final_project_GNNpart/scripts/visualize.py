# scripts/visualize.py
"""
Visualize incremental training results for METR-LA and PEMS-Bay.
Generates one bar chart per dataset, facet by time frequency,
models on x-axis, adjacency variants as hue.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Compatibility patch for matplotlib colormap registration
import matplotlib as mpl
import matplotlib.cm as cm
if not hasattr(cm, 'register_cmap'):
    cm.register_cmap = mpl.colormaps.register

# Mapping from stride to human-readable time frequency
TIMEFREQ_MAP = {1: '5min', 12: '1hour', 288: '1day'}
# Order of models on x-axis
MODEL_ORDER = ['baseline', 'gcn', 'gat', 'cheb', 'appnp', 'sgc']
# Adjacency variants (key in CSV "dataset" column) and display labels
ADJ_VARIANTS = [
    ('adj_mx_knn',         'KNN'),
    ('adj_mx_knn_lane',    'KNN+Lane'),
    ('adj_mx_radius',      'Radius'),
    ('adj_mx_radius_lane', 'Radius+Lane'),
]
PALETTE = sns.color_palette('Set2', n_colors=len(ADJ_VARIANTS))
ADJ_KEYS   = [k for k, _ in ADJ_VARIANTS]
ADJ_LABELS = [l for _, l in ADJ_VARIANTS]


def collect_incremental_results(input_dir):
    """
    Read all CSV files in input_dir matching pattern:
      results_<dataset>_<model>_incremental_rmse.csv
    Returns DataFrame with columns:
      dataset_type, adj, model, stride, timefreq, rmse_mph
    """
    records = []
    pattern = os.path.join(input_dir, 'results_*_incremental_rmse.csv')
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        parts = fname.split('_')  # e.g. ['results','metrla','gcn','incremental','rmse.csv']
        if len(parts) < 5:
            continue
        ds_type = parts[1]  # 'metrla' or 'pems'
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            full_adj = row['dataset']
            # strip dataset suffix
            suffix = f"_{ds_type}"
            adj_key = full_adj[:-len(suffix)] if full_adj.endswith(suffix) else full_adj
            stride   = int(row['stride'])
            model    = row['model']
            rmse     = float(row['rmse_mph'])
            timefreq = TIMEFREQ_MAP.get(stride, str(stride))
            records.append({
                'dataset_type': ds_type,
                'adj':          adj_key,
                'model':        model,
                'stride':       stride,
                'timefreq':     timefreq,
                'rmse_mph':     rmse
            })
    if records:
        return pd.DataFrame(records)
    cols = ['dataset_type','adj','model','stride','timefreq','rmse_mph']
    return pd.DataFrame(columns=cols)


def plot_adj_comparison(df, dataset_type, out_file=None):
    """
    Plot bar charts for a single dataset, faceted by time frequency.
    """
    sns.set(style='whitegrid')

    # Filter only supported adjacency matrix variants
    df = df[df['adj'].isin(ADJ_KEYS)].copy()

    # Map raw adj keys to readable labels
    df['adjacency matrix'] = df['adj'].map(dict(ADJ_VARIANTS))

    # Set categorical order for models and time frequencies
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    df['timefreq'] = pd.Categorical(df['timefreq'], categories=['5min', '1hour', '1day'], ordered=True)

    # Create a bar plot with Seaborn's FacetGrid
    g = sns.catplot(
        data=df,
        x='model', y='rmse_mph',
        hue='adjacency matrix', hue_order=ADJ_LABELS,
        palette=PALETTE,
        col='timefreq',
        kind='bar',
        height=4, aspect=1.2,
    )

    # Set axis labels and subplot titles
    g.set_axis_labels('Model', 'Test RMSE (mph)')
    g.set_titles(col_template='{col_name}', size=18)

    # 🔥 Remove default legend added by Seaborn
    if g._legend:
        g._legend.remove()

    # Create custom legend handles (color patches)
    handles = [
        Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=label)
        for label, color in zip(ADJ_LABELS, PALETTE)
    ]
    labels = ADJ_LABELS

    # Add a single global legend to the figure
    fig = g.fig
    fig.legend(
        handles, labels,
        title='',
        loc='center left', bbox_to_anchor=(0.9, 0.5),
        frameon=False,
        prop={'size': 16},       # Increase legend font size
        borderaxespad=0.5,
        labelspacing=1.2         # Space between legend entries
    )

    # Increase x-axis tick font size for all subplots
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', labelsize=18)

    # Adjust space for legend
    fig.subplots_adjust(right=0.9)

    # Add a main title for the entire figure
    fig.suptitle(f'{dataset_type.upper()} - Incremental Training', y=1.02, size=18)

    # Save figure to file if specified
    if out_file:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')

    # Show the plot
    plt.show()


def main():
    input_dir = 'output_incremental'
    df_inc = collect_incremental_results(input_dir)
    if df_inc.empty:
        print(f'No incremental results found in {input_dir}')
        return
    for ds in df_inc['dataset_type'].unique():
        df_ds = df_inc[df_inc['dataset_type'] == ds]
        out_file = f'incremental_{ds}_adj_comparison.png'
        plot_adj_comparison(df_ds, ds, out_file=out_file)
        print(f'Saved plot for {ds} to {out_file}')


if __name__ == '__main__':
    main()
