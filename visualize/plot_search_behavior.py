import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("husl", n_colors=2)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-base-f1-wScore.csv',
    'visualize/wandb/searchr1-instruct-may6.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'AutoRefine-Base',
    'Search-R1-Base',
    # 'AutoRefine-Instruct',
    # 'Search-R1-Instruct',
    'y=1',
]

col_names = [
    "on training samples",
    "on all 7 benchmarks",
    "on 3 single-hop benchmarks",
    "on 4 multi-hop benchmarks",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
axes = axes.flatten()

selected_columns = [
    "env/number_of_valid_search",
    "val/number_of_valid_search/mean",
    "val/number_of_valid_search/single",
    "val/number_of_valid_search/multi",
]

y_labels = [
    '# Search Calls',
    '# Search Calls',
    '# Search Calls',
    '# Search Calls',
]

max_x = 181
min_x = 1
max_y = [1.4, 1.6, 1.3, 1.8]
min_y = [0.8, 0.8, 0.8, 0.8]
for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>10 else None
    l1, = axes[i].plot(exp_dfs[0][col][min_x:max_x].dropna(), label=exp_names[0], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i].plot(exp_dfs[1][col][min_x:max_x].dropna(), label=exp_names[1], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=4)
    l3 = axes[i].axhline(y=1.0, color='orange', linewidth=2, linestyle='--', label=exp_names[-1], zorder=3)
    if i == 0:
        lines = [l1, l2, l3]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i].set_xlabel('Training Steps', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    axes[i].set_title(col_name, fontsize=12) #, weight='bold')
    axes[i].set_xlim(-5, max_x+5)
    axes[i].set_ylim(min_y[i] -.03, max_y[i]+.03)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[i].grid(True)
fig.legend(handles=lines, loc='center', ncol=3, bbox_to_anchor=(0.5, 0.53))

selected_columns = [
    "critic/info_score/mean",
    "val/information_scores/mean",
    "val/information_scores/single",
    "val/information_scores/multi",
]

y_labels = [
    'Success Rate (%)',
    'Success Rate (%)',
    'Success Rate (%)',
    'Success Rate (%)',
]
max_y = [.7, .6, .75, .55]
min_y = [.2, .2, .3, .15]
scale=100

for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>10 else None
    l1, = axes[i+4].plot(exp_dfs[0][col][min_x:max_x].dropna()*scale, label=exp_names[0], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i+4].plot(exp_dfs[1][col][min_x:max_x].dropna()*scale, label=exp_names[1], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=4)
    if i == 0:
        lines = [l1, l2]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i+4].set_xlabel('Training Steps', fontsize=12)
    axes[i+4].set_ylabel(y_labels[i], fontsize=12)
    axes[i+4].set_title(col_name, fontsize=12) #, weight='bold')
    axes[i].set_xlim(-5, max_x+5)
    axes[i+4].set_ylim((min_y[i]-.03)*scale, (max_y[i]+.03)*scale)
    axes[i+4].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[i+4].grid(True)

fig.text(0.5, 0.96, '(a) Search Frequency', ha='center', va='bottom', fontsize=16, weight='bold')
fig.text(0.5, 0.44, '(b) Search Quality', ha='center', va='bottom', fontsize=16, weight='bold')
fig.legend(handles=lines, loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(top=0.91, bottom=0.12, hspace=0.8)
plt.savefig('visualize/figures/curve_search_behavior.pdf', bbox_inches='tight', dpi=300)
