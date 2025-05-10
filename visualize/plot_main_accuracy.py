import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("Set2", n_colors=3)

colors = [
    line_palette[0],
    line_palette[1],
    line_palette[2],
]

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-base-f1-wScore.csv',
    'visualize/wandb/searchr1-base-may8.csv',
    'visualize/wandb/research-base-may9.csv',
    'visualize/wandb/searchr1-base_2.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'AutoRefine-Base',
    'Search-R1-Base',
    'Research-Base',
]

selected_columns = [
    "val/test_score/mean",
    # "val/test_score/hotpotqa",
    "val/test_score/2wikimultihopqa",
    "val/test_score/musique",
    "val/test_score/bamboogle",
]
y_labels = ['Accuracy'] + [''] * len(selected_columns)
col_names = [
    "(a) All 7 Datasets",
    "(b) 2WikiMultihopQA",
    "(c) Musique",
    "(d) Bamboogle",
]

y_lims = [
    (.0, .5),
    (.0, .5),
    (.0, .2),
    (.0, .5),
]
y_lims = [(i-.03, j+.03) for i, j in y_lims]
max_x=165
min_x=1
    
# Reorder plot indices: move (b) to bottom left (index 3)
plot_order = [0, 1, 2, 3]  # original indices of selected_columns

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
axes = axes.flatten()

# Plotting loop with reordered subplot placement
for plot_idx, original_idx in enumerate(plot_order):
    col = selected_columns[original_idx]
    col_name = col_names[original_idx]
    marker = '^' if original_idx >-1 else None
    l1, = axes[plot_idx].plot(exp_dfs[0][col][min_x:max_x].dropna(), label=exp_names[0], alpha=1, color=colors[0], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[plot_idx].plot(exp_dfs[1][col][min_x:max_x].dropna(), label=exp_names[1], alpha=1, color=colors[1], marker=marker, linestyle='-', linewidth=2, zorder=4)
    l3, = axes[plot_idx].plot(exp_dfs[2][col][min_x:max_x].dropna(), label=exp_names[2], alpha=1, color=colors[2], marker=marker, linestyle='-', linewidth=2, zorder=3)
    
    if original_idx == 0:
        lines = [l1, l2, l3]
        labels = exp_names

    axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
    axes[plot_idx].set_ylabel(y_labels[original_idx], fontsize=12)
    axes[plot_idx].set_title(col_name, fontsize=12, weight='bold')
    axes[plot_idx].set_xlim(-5, max_x+2)
    axes[plot_idx].set_ylim(*y_lims[original_idx])
    axes[plot_idx].set_xticks(np.arange(0, max_x+1, 25))
    # axes[plot_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[plot_idx].grid(True)


# Legend and save
fig.legend(handles=lines, loc='lower center', ncol=3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.26, wspace=0.15)
plt.savefig('visualize/figures/curve_main_accuracy.pdf', bbox_inches='tight', dpi=300)
