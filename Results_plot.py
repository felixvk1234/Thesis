import numpy as np
import pandas as pd

# Re-define the datasets with GAT-Short and GAT-Long added manually
datasets = {
    "Mobile Vikings (Imbalance 8.3%)": {
        "lift_05": pd.DataFrame({
            'None': [2.86, 2.75, 2.05, 2.70, 0.76, 0.50, 0.32, 0.49],
            'Duration': [3.64, 0.89, np.nan, np.nan, 2.79, 0.50, np.nan, np.nan],
            'Count': [3.52, 3.36, np.nan, np.nan, 3.05, 4.56, np.nan, np.nan],
            'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
        }).set_index('index'),

        "lift_5": pd.DataFrame({
            'None': [2.97, 2.74, 3.26, 3.24, 1.15, 1.16, 0.60, 0.66],
            'Duration': [2.88, 0.67, np.nan, np.nan, 1.95, 2.27, np.nan, np.nan],
            'Count': [2.68, 1.66, np.nan, np.nan, 2.91, 3.92, np.nan, np.nan],
            'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
        }).set_index('index'),

        "auc": pd.DataFrame({
            'None': [0.80, 0.64, 0.59, 0.56, 0.64, 0.65, 0.58, 0.56],
            'Duration': [0.81, 0.54, np.nan, np.nan, 0.69, 0.62, np.nan, np.nan],
            'Count': [0.81, 0.77, np.nan, np.nan, 0.77, 0.74, np.nan, np.nan],
            'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
        }).set_index('index'),

        "emp": pd.DataFrame({
            'None': [8, 5, 6, 4, 6, 5, 6, 5],
            'Duration': [8, 4, np.nan, np.nan, 6, 4, np.nan, np.nan],
            'Count': [8, 6, np.nan, np.nan, 7, 6, np.nan, np.nan],
            'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
        }).set_index('index'),

        "auprc": pd.DataFrame({
            'None': [0.536, 0.639, 0.462, 0.382, 0.295, 0.273, 0.248, 0.221],
            'Duration': [0.57, 0.21, np.nan, np.nan, 0.37, 0.35, np.nan, np.nan],
            'Count': [0.55, 0.43, np.nan, np.nan, 0.49, 0.56, np.nan, np.nan],
            'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
        }).set_index('index'),
    }
}

datasets["Proximus Pre (Imbalance 4.4%)"] = {
    "lift_05": pd.DataFrame({
        'None': [0.12, 0.65, 1.11, 0.17, 0.46, 0.67, 0.24, 0.12],
        'Duration': [0, 1.54, np.nan, np.nan, 0.15, 0.47, np.nan, np.nan],
        'Count': [0.21, 0.35, np.nan, np.nan, 0.43, 0.53, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "lift_5": pd.DataFrame({
        'None': [0.16, 0.77, 1.31, 0.22, 0.30, 0.70, 0.24, 0.12],
        'Duration': [0.24, 1.59, np.nan, np.nan, 0.20, 0.70, np.nan, np.nan],
        'Count': [0.19, 0.92, np.nan, np.nan, 0.24, 0.42, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "auc": pd.DataFrame({
        'None': [0.42, 0.50, 0.61, 0.33, 0.32, 0.52, 0.43, 0.39],
        'Duration': [0.40, 0.67, np.nan, np.nan, 0.44, 0.43, np.nan, np.nan],
        'Count': [0.35, 0.62, np.nan, np.nan, 0.40, 0.34, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "emp": pd.DataFrame({
        'None': [np.nan] * 8,
        'Duration': [np.nan] * 8,
        'Count': [np.nan] * 8,
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "auprc": pd.DataFrame({
        'None': [0.035, 0.041, 0.054, 0.030, 0.030, 0.044, 0.036, 0.032],
        'Duration': [0.034, 0.421, np.nan, np.nan, 0.036, 0.035, np.nan, np.nan],
        'Count': [0.031, 0.056, np.nan, np.nan, 0.034, 0.030, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),
}

datasets["Proximus Post (Imbalance 0.84%)"] = {
    "lift_05": pd.DataFrame({
        'None': [0.10, 3.03, 1.83, 2.77, 4.76, 6.06, 1.61, 1.08],
        'Duration': [0, 0.12, np.nan, np.nan, 5.43, 0.51, np.nan, np.nan],
        'Count': [0.12, 1.48, np.nan, np.nan, 0.43, 0.53, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "lift_5": pd.DataFrame({
        'None': [0.10, 6.08, 5.03, 1.99, 4.36, 6.10, 4.32, 1.17],
        'Duration': [0.30, 0.77, np.nan, np.nan, 4.49, 1.21, np.nan, np.nan],
        'Count': [0.19, 2.17, np.nan, np.nan, 0.47, 0.41, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "auc": pd.DataFrame({
        'None': [0.24, 0.85, 0.86, 0.43, 0.83, 0.82, 0.86, 0.79],
        'Duration': [0.68, 0.74, np.nan, np.nan, 0.83, 0.78, np.nan, np.nan],
        'Count': [0.54,0.84, np.nan, np.nan, 0.72, 0.22, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "emp": pd.DataFrame({
        'None': [np.nan] * 8,
        'Duration': [np.nan] * 8,
        'Count': [np.nan] * 8,
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),

    "auprc": pd.DataFrame({
        'None': [0.005, 0.038, 0.038, 0.010, 0.033, 0.040, 0.035, 0.020],
        'Duration': [0.012, 0.017, np.nan, np.nan, 0.033, 0.024, np.nan, np.nan],
        'Count': [0.008, 0.028, np.nan, np.nan, 0.014, 0.005, np.nan, np.nan],
        'index': ['GCN-Short', 'GCN-Long', 'GraphSAGE-Short', 'GraphSAGE-Long', 'GAT-Short', 'GAT-Long','GIN-Short','GIN-Long']
    }).set_index('index'),
}

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams

# Set better default fonts
rcParams['font.family'] = 'DejaVu Sans'

def plot_heatmap(ax, data, vmin, vmax, cmap='YlOrRd', fmt=".2f"):
    """Improved heatmap with guaranteed visible annotations"""
    mask = data.isnull()
    plot_data = data.copy()
    plot_data.index.name = None
    plot_data.columns.name = None

    # Create base heatmap
    sns.heatmap(plot_data, mask=mask, cbar=False, ax=ax,
                cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='black',
                square=False, annot=False)

    # Add annotations
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            if not mask.iloc[i,j]:
                value = plot_data.iloc[i,j]
                ax.text(j + 0.5, i + 0.5, f"{value:{fmt}}",
                        ha="center", va="center",
                        color="black", fontsize=8)

    # Formatting
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)

    ax.set_xticks(np.arange(plot_data.shape[1]) + 0.5)
    ax.set_xticklabels(plot_data.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
    ax.set_yticklabels(plot_data.index, rotation=0, ha='right', fontsize=8)

# Define your custom ranges for each dataset-metric combination
range_config = {
    "Mobile Vikings (Imbalance 8.3%)": {
        "lift_05": {"vmin": 0, "vmax": 4, "fmt": ".2f"},
        "lift_5": {"vmin": 0, "vmax": 4, "fmt": ".2f"},
        "auc": {"vmin": 0.45, "vmax": 0.85, "fmt": ".2f"},
        "emp": {"vmin": 0, "vmax": 8, "fmt": ".0f"},
        "auprc": {"vmin": 0.2, "vmax": 0.65, "fmt": ".3f"},
    },
    "Proximus Pre (Imbalance 4.4%)": {
        "lift_05": {"vmin": 0, "vmax": 2, "fmt": ".2f"},
        "lift_5": {"vmin": 0, "vmax": 2, "fmt": ".2f"},
        "auc": {"vmin": 0.3, "vmax": 0.7, "fmt": ".2f"},
        "emp": {"vmin": 0, "vmax": 1, "fmt": ".0f"},  # Not used but needs a range
        "auprc": {"vmin": 0, "vmax": 0.45, "fmt": ".3f"},
    },
    "Proximus Post (Imbalance 0.84%)": {
        "lift_05": {"vmin": 0, "vmax": 4, "fmt": ".2f"},
        "lift_5": {"vmin": 0, "vmax": 7, "fmt": ".2f"},
        "auc": {"vmin": 0.2, "vmax": 0.9, "fmt": ".2f"},
        "emp": {"vmin": 0, "vmax": 1, "fmt": ".0f"},  # Not used but needs a range
        "auprc": {"vmin": 0, "vmax": 0.04, "fmt": ".3f"},
    }
}

# Create figure
fig, axs = plt.subplots(
    len(range_config[list(datasets.keys())[0]]),  # Number of metrics
    len(datasets),                                # Number of datasets
    figsize=(12, 15),
    dpi=300,
    gridspec_kw={'hspace': 0.3, 'wspace': 0.1}
)

# Plot each dataset and metric with individual ranges
for col_idx, dataset_name in enumerate(datasets.keys()):
    for row_idx, metric_key in enumerate(range_config[dataset_name].keys()):
        ax = axs[row_idx, col_idx]
        data = datasets[dataset_name].get(metric_key)
        config = range_config[dataset_name][metric_key]

        if data is not None and not data.isnull().all().all():
            plot_heatmap(ax, data,
                         vmin=config["vmin"],
                         vmax=config["vmax"],
                         fmt=config["fmt"])

            # Axis label handling
            if col_idx != 0:
                ax.set_yticklabels([])
                ax.set_ylabel('')
            if row_idx != len(range_config[dataset_name]) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
        else:
            ax.axis('off')

# Add titles
metric_names = {
    "lift_05": "0.5% Lift",
    "lift_5": "5% Lift",
    "auc": "AUC",
    "emp": "EMP",
    "auprc": "AUPRC"
}

# Dataset names at top
for ax, dataset_name in zip(axs[0, :], datasets.keys()):
    ax.set_title(dataset_name, size=10, pad=10)

# Metric names at left
for row_idx, metric_key in enumerate(range_config[list(datasets.keys())[0]].keys()):
    axs[row_idx, 0].set_ylabel(metric_names[metric_key],
                              rotation=0, labelpad=60,
                              size=10, va='center')

plt.tight_layout()
plt.show()
