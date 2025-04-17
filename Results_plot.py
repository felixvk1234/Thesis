import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define the four DataFrames based on the image content

# 0.5% Lift
lift_05 = pd.DataFrame({
    'None': [2.86, 2.75,2.05,2.70],
    'Duration': [3.64, 0.89,'none','none' ],
    'Count': [3.52, 3.36,'none','none'],
    'Duration + Count': [1.23, 1.06,'none','none'],
    'index': ['GCN-Short', 'GCN-Long','GraphSAGE-Short', 'GraphSAGE-Long']
}).set_index('index')

# 5% Lift
lift_5 = pd.DataFrame({
    'None': [2.97, 2.74,3.26,3.24],
    'Duration': [2.88, 0.67,'none','none'],
    'Count': [2.68, 1.66,'none','none'],
    'Duration + Count': [1.90, 0.89,'none','none'],
    'index': ['GCN-Short', 'GCN-Long','GraphSAGE-Short', 'GraphSAGE-Long']
}).set_index('index')

# AUC
auc = pd.DataFrame({
    'None': [0.80, 0.64,0.59,0.56],
    'Duration': [0.81, 0.54,'none','none'],
    'Count': [0.81, 0.77,'none','none'],
    'Duration + Count': [0.81, 0.53,'none','none'],
    'index': ['GCN-Short', 'GCN-Long','GraphSAGE-Short', 'GraphSAGE-Long']
}).set_index('index')

# EMP
emp = pd.DataFrame({
    'None': [8, 5,6,3],
    'Duration': [8, 4,'none','none'],
    'Count': [8, 6,'none','none'],
    'Duration + Count': [8, 5, 'none','none'],
    'index': ['GCN-Short', 'GCN-Long','GraphSAGE-Short', 'GraphSAGE-Long']
}).set_index('index')

# Replace 'none' with np.nan in all DataFrames
lift_05.replace('none', np.nan, inplace=True)
lift_5.replace('none', np.nan, inplace=True)
auc.replace('none', np.nan, inplace=True)
emp.replace('none', np.nan, inplace=True)

def plot_heatmap(ax, data, title, vmin=None, vmax=None, cmap='Oranges', fmt=".2f"):
    sns.heatmap(data, annot=True, fmt=fmt, cbar=False, ax=ax,
                annot_kws={"size": 8}, linewidths=0.5, linecolor='black',
                cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', which='major', labelsize=8, labelrotation=45)
    ax.tick_params(axis='y', which='major', labelsize=8, labelrotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1)

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot each heatmap with its own meaningful color scale
plot_heatmap(axs[0, 0], lift_05, "0.5% Lift", vmin=0, vmax=4, fmt=".2f")
plot_heatmap(axs[0, 1], lift_5, "5% Lift", vmin=0, vmax=4, fmt=".2f")
plot_heatmap(axs[1, 0], auc, "AUC", vmin=0.50, vmax=0.82, fmt=".2f")
plot_heatmap(axs[1, 1], emp, "EMP", vmin=3, vmax=9, fmt=".0f")

plt.tight_layout()
plt.show()


