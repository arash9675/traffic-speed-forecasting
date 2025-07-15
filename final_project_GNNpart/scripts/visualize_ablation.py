import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Mapping from stride to human-readable time frequency
TIMEFREQ_MAP = {1: '5min', 12: '1hour', 288: '1day'}
# Adjacency variants (key in CSV "dataset" column) and display labels
ADJ_VARIANTS = [
    ('adj_mx_knn',         'KNN'),
    ('adj_mx_knn_lane',    'KNN+Lane'),
    ('adj_mx_radius',      'Radius'),
    ('adj_mx_radius_lane', 'Radius+Lane'),
]
PALETTE   = sns.color_palette('Set2', n_colors=len(ADJ_VARIANTS))
ADJ_KEYS  = [k for k, _ in ADJ_VARIANTS]
ADJ_LABELS= [l for _, l in ADJ_VARIANTS]

MODEL = 'gcn'
TRAINING_ORDER = ['incremental', 'cumulative']


def collect_gcn_results(cumu_dir, incr_dir):
    records = []
    for training, input_dir in [('cumulative', cumu_dir),
                                 ('incremental', incr_dir)]:
        pattern = os.path.join(input_dir, f'results_*_{MODEL}_{training}_rmse.csv')
        for path in glob.glob(pattern):
            ds_type = os.path.basename(path).split('_')[1]
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                full_adj = row['dataset']
                suffix = f"_{ds_type}"
                adj_key = full_adj[:-len(suffix)] if full_adj.endswith(suffix) else full_adj
                if adj_key not in ADJ_KEYS:
                    continue
                stride = int(row['stride'])
                rmse   = float(row['rmse_mph'])
                timefreq = TIMEFREQ_MAP.get(stride, str(stride))
                records.append({
                    'dataset_type': ds_type,
                    'adj':          adj_key,
                    'stride':       stride,
                    'timefreq':     timefreq,
                    'rmse_mph':     rmse,
                    'training':     training
                })
    return pd.DataFrame(records)


def plot_gcn_comparison(df, dataset_type, out_file=None):
    sns.set(style='whitegrid')
    df = df[df['dataset_type'] == dataset_type].copy()
    if df.empty:
        print(f'No data for {dataset_type}')
        return

    df['training'] = pd.Categorical(df['training'],
                                    categories=TRAINING_ORDER, ordered=True)
    df['timefreq'] = pd.Categorical(df['timefreq'],
                                    categories=['5min','1hour','1day'],
                                    ordered=True)
    df['adj_label'] = df['adj'].map(dict(ADJ_VARIANTS))

    g = sns.catplot(
        data=df,
        x='training', y='rmse_mph',
        hue='adj_label', hue_order=ADJ_LABELS,
        palette=PALETTE,
        col='timefreq',
        kind='bar',
        height=4, aspect=1.2,
    )

    g.set_axis_labels('Training Mode', 'Test RMSE (mph)')
    g.set_titles(col_template='{col_name}', size=16)
    if g._legend:
        g._legend.remove()

    handles = [
        Patch(facecolor=col, edgecolor='black', lw=0.5, label=lab)
        for lab, col in zip(ADJ_LABELS, PALETTE)
    ]
    g.fig.legend(
        handles, ADJ_LABELS,
        title='Adjacency',
        loc='center left', bbox_to_anchor=(0.9, 0.5),
        frameon=False, prop={'size':14}
    )

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=14)

    g.fig.subplots_adjust(right=0.88)
    g.fig.suptitle(f'GCN: {dataset_type.upper()} Cumulative vs Incremental', y=1.02, size=18)

    if out_file:
        g.fig.savefig(out_file, dpi=200, bbox_inches='tight')
        print(f'Saved plot for {dataset_type} → {out_file}')
    else:
        plt.show()


def main():
    cumu_dir = 'output_cumulative'
    incr_dir = 'output_incremental'
    df_all = collect_gcn_results(cumu_dir, incr_dir)
    if df_all.empty:
        print('No GCN results found.')
        return
    for ds in ['metrla', 'pems']:
        plot_gcn_comparison(df_all, ds,
                            out_file=f'gcn_{ds}_incr_vs_cumu.png')

if __name__ == '__main__':
    main()
