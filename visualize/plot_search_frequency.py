import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("husl", n_colors=2)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-base-wScore-2.csv',
    'visualize/wandb/ours-instruct-wScore.csv',
    'visualize/wandb/searchr1-base_2.csv',
    'visualize/wandb/searchr1-instruct.csv',
    'visualize/wandb/ours-base.csv',
    # 'visualize/wandb/ours-instruct.csv',
    # 'visualize/wandb/searchr1-base_2.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]

exp_names = [
    'AutoRefine-Base',
    'AutoRefine-Instruct',
    'Search-R1-Base',
    'Search-R1-Instruct',
]

selected_columns = [
    "env/number_of_valid_search",
    "easy/env/number_of_valid_search",
    "hard/env/number_of_valid_search",
]

exp_dfs[3][selected_columns[0]] = -exp_dfs[2][selected_columns[0]]


y_labels = [
    'Number of Search Calls',
    'Number of Search Calls',
    'Number of Search Calls',
    'Number of Search Calls',
]
col_names = [
    '(a) Base v.s. Instruct Models',
    '(b) Easy v.s. Hard Samples',
]
    
max_x = 199
y_lim = [
    (0.5, 2.1),
    (0.5, 2.1),
    (0.5, 2.1),
    (0.5, 2.1),
]
y_ticks = [
    np.arange(0.7, 2.1, 0.3),
    np.arange(0.7, 2.1, 0.3),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
axes = axes.flatten()

idx = 0
exp_id = [0, 2]
linestyle = '-'
col = selected_columns[0]
l1, = axes[idx].plot(exp_dfs[exp_id[0]][col][:max_x].dropna(), label=exp_names[exp_id[0]], alpha=1, color=line_palette[1], linestyle=linestyle, linewidth=2, zorder=6)
l2, = axes[idx].plot(exp_dfs[exp_id[1]][col][:max_x].dropna(), label=exp_names[exp_id[1]], alpha=1, color=line_palette[0], linestyle=linestyle, linewidth=2, zorder=5)


idx = 1
exp_id = [1, 3]
linestyle = '--'
col = selected_columns[0]
l3, = axes[idx].plot(exp_dfs[exp_id[0]][col][:max_x].dropna(), label=exp_names[exp_id[0]], alpha=1, color=line_palette[1], linestyle=linestyle, linewidth=2, zorder=6)
l4, = axes[idx].plot(exp_dfs[exp_id[1]][col][:max_x].dropna(), label=exp_names[exp_id[1]], alpha=1, color=line_palette[0], linestyle=linestyle, linewidth=2, zorder=5)


idx = 2
exp_id = [0, 2]
linestyle = '-'
col = selected_columns[1]
_, = axes[idx].plot(exp_dfs[exp_id[0]][col][:max_x].dropna(), label=exp_names[exp_id[0]], alpha=1, color=line_palette[1], linestyle=linestyle, linewidth=2, zorder=6)
_, = axes[idx].plot(exp_dfs[exp_id[1]][col][:max_x].dropna(), label=exp_names[exp_id[1]], alpha=1, color=line_palette[0], linestyle=linestyle, linewidth=2, zorder=5)


idx = 3
exp_id = [0, 2]
linestyle = '-'
col = selected_columns[2]
_, = axes[idx].plot(exp_dfs[exp_id[0]][col][:max_x].dropna(), label=exp_names[exp_id[0]], alpha=1, color=line_palette[1], linestyle=linestyle, linewidth=2, zorder=6)
_, = axes[idx].plot(exp_dfs[exp_id[1]][col][:max_x].dropna(), label=exp_names[exp_id[1]], alpha=1, color=line_palette[0], linestyle=linestyle, linewidth=2, zorder=5)


lines = [l1, l3, l2, l4]
labels = exp_names


for i in range(4):
    axes[i].set_xlabel('Training Steps', fontsize=10)
    if y_labels[i]:
        axes[i].set_ylabel(y_labels[i], fontsize=10)
    # axes[i].set_title(y_titles[i], fontsize=12, weight='bold')
    axes[i].set_xlim(-5, max_x+5)
    axes[i].set_ylim(*y_lim[i])
    # axes[i].set_yticks(y_ticks[i])
    axes[i].grid(True)

fig.text(0.26, 0.95, col_names[0], ha='center', va='center', fontsize=12, weight='bold')
fig.text(0.76, 0.95, col_names[1], ha='center', va='center', fontsize=12, weight='bold')
fig.legend(handles=lines, loc='lower center', ncol=4)

plt.tight_layout()
plt.subplots_adjust(bottom=0.23, top=0.9)
plt.savefig('visualize/figures/curve_search_frequency.pdf', bbox_inches='tight', dpi=300)