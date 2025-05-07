import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("husl", n_colors=2)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-base-wScore-2.csv',
    'visualize/wandb/searchr1-base_2.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'AutoRefine-Base',
    'Search-R1-Base',
    # 'AutoRefine-Instruct',
    # 'Search-R1-Instruct',
]

col_names = [
    "on Training Samples",
    "on all 7 Benchmarks",
    "on 3 Single-Hop Benchmarks",
    "on 4 Multi-Hop Benchmarks",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
axes = axes.flatten()

selected_columns = [
    "env/number_of_valid_search",
    "env/number_of_valid_search",
    "easy/env/number_of_valid_search",
    "hard/env/number_of_valid_search",
]

y_labels = [
    '# Search Calls',
    '# Search Calls',
    '# Search Calls',
    '# Search Calls',
]

max_y = [1.6, 1.6, 1.6, 1.6]
min_y = [0.5, 0.5, 0.5, 0.5]
for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>10 else None
    l1, = axes[i].plot(exp_dfs[0][col][:201].dropna(), label=exp_names[0], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i].plot(exp_dfs[1][col][:201].dropna(), label=exp_names[1], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=4)
    if i == 0:
        lines = [l1, l2]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i].set_xlabel('Training Steps', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    axes[i].set_title(col_name, fontsize=12) #, weight='bold')
    axes[i].set_xlim(-5, 210)
    axes[i].set_ylim(min_y[i] -.03, max_y[i]+.03)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[i].grid(True)


selected_columns = [
    "critic/info_score/mean",
    "critic/info_score/mean",
    "easy/critic/info_score/mean",
    "hard/critic/info_score/mean",
]

y_labels = [
    'Success Rate (%)',
    'Success Rate (%)',
    'Success Rate (%)',
    'Success Rate (%)',
]

for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>10 else None
    l1, = axes[i+4].plot(exp_dfs[0][col][:201].dropna(), label=exp_names[0], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i+4].plot(exp_dfs[1][col][:201].dropna(), label=exp_names[1], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=4)
    if i == 0:
        lines = [l1, l2]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i+4].set_xlabel('Training Steps', fontsize=12)
    axes[i+4].set_ylabel(y_labels[i], fontsize=12)
    axes[i+4].set_title(col_name, fontsize=12) #, weight='bold')
    axes[i+4].set_xlim(-5, 210)
    # axes[i+4].set_ylim(0, max_y[i]+.03)
    # axes[i+4].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[i+4].grid(True)

fig.text(0.5, 0.96, 'Search Frequency', ha='center', va='bottom', fontsize=14, weight='bold')
fig.text(0.5, 0.48, 'Search Quality', ha='center', va='bottom', fontsize=14, weight='bold')
fig.legend(handles=[l1, l2], loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(top=0.91, bottom=0.12, hspace=0.53)
plt.savefig('visualize/figures/curve_search_behavior.pdf', bbox_inches='tight', dpi=300)
