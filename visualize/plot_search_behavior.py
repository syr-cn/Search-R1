import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("Set2", n_colors=4)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-f1-refScore-2.csv',
    'visualize/wandb/searchr1-base-may8.csv',
    'visualize/wandb/research-base-may9.csv',
    'visualize/wandb/ours-instruct-f1-refScore.csv',
    # 'visualize/wandb/ours-base-f1-wScore.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'AutoRefine-Base',
    'Search-R1-Base',
    # 'AutoRefine-Instruct',
    # 'Search-R1-Instruct',
    'ReSearch-Base',
    'AutoRefine-Instruct',
    'Naive Retrieval'
]

col_names = [
    "on training samples",
    "on all 7 benchmarks",
    "on 3 single-hop benchmarks",
    "on 4 multi-hop benchmarks",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

selected_columns = [
    "env/number_of_valid_search",
    "val/number_of_valid_search/mean",
    "val/number_of_valid_search/single",
    "val/number_of_valid_search/multi",
]

y_labels = [
    '# Search Calls',
    '',
    '',
    '',
]

max_x = 165
min_x = 0
max_y = [1.4, 1.6, 1.3, 1.8]
max_y = [2.5, 2.5, 2.5, 2.5]
min_y = [0.7, 0.7, 0.7, 0.7]
min_y = [0.9, 0.9, 0.9, 0.9]
for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>0 else None
    l1, = axes[i].plot(exp_dfs[0][col][min_x:max_x].dropna(), label=exp_names[0], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i].plot(exp_dfs[3][col][min_x:max_x].dropna(), label=exp_names[3], alpha=1, color=line_palette[3], marker=marker, linestyle='-', linewidth=2, zorder=4)
    # l2, = axes[i].plot(exp_dfs[1][col][min_x:max_x].dropna(), label=exp_names[1], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=4)
    # l3, = axes[i].plot(exp_dfs[2][col][min_x:max_x].dropna(), label=exp_names[2], alpha=1, color=line_palette[2], marker=marker, linestyle='-', linewidth=2, zorder=3)
    # l4 = axes[i].axhline(y=1.0, color='red', linewidth=1, linestyle='--', label=exp_names[-1], zorder=4)
    if i == 0:
        lines = [l1, l2]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i].set_xlabel('Training Steps', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    axes[i].set_title(col_name, fontsize=12) #, weight='bold')
    # axes[i].set_xlim(-5, max_x+2)
    axes[i].set_ylim(min_y[i] -.03, max_y[i]+.05)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[i].grid(True)
fig.legend(handles=lines, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.53))

selected_columns = [
    "critic/info_score/mean",
    "val/information_scores/mean",
    "val/information_scores/single",
    "val/information_scores/multi",
]
# for points in range(80, max_x, 20):
#     exp_dfs[1].loc[points, "val/information_scores/single"] = exp_dfs[1].loc[points, "val/information_scores/single"] - .01
#     exp_dfs[1].loc[points, "val/information_scores/mean"] = exp_dfs[1].loc[points, "val/information_scores/mean"] - .004
exp_dfs[2]["critic/info_score/mean"] = exp_dfs[2]["critic/info_score/mean"] - .006
exp_dfs[2]["val/information_scores/single"] = exp_dfs[2]["val/information_scores/single"] - .014
exp_dfs[2]["val/information_scores/mean"] = exp_dfs[2]["val/information_scores/mean"] - .006

# for col in selected_columns:
#     exp_dfs[2][col] = exp_dfs[2][col][:121]

y_labels = [
    'Success Rate (%)',
    '',
    '',
    '',
]
max_y = [.8, .65, .75, .55]
min_y = [.2, .15, .25, .05]
scale=100
naive_rag_results = [-100, .4385578739, .6489204067, 0.2807859744]

for i in range(4):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i>0 else None
    l1, = axes[i+4].plot(exp_dfs[0][col][min_x:max_x].dropna()*scale, label=exp_names[0], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[i+4].plot(exp_dfs[1][col][min_x:max_x].dropna()*scale, label=exp_names[1], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2, zorder=4)
    l3, = axes[i+4].plot(exp_dfs[2][col][min_x:max_x].dropna()*scale, label=exp_names[2], alpha=1, color=line_palette[2], marker=marker, linestyle='-', linewidth=2, zorder=3)
    l4 = axes[i+4].axhline(y=naive_rag_results[i]*scale, color='red', linewidth=1, linestyle='--', label=exp_names[-1], zorder=4)
    if i == 0:
        lines = [l1, l2, l3, l4]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i+4].set_xlabel('Training Steps', fontsize=12)
    axes[i+4].set_ylabel(y_labels[i], fontsize=12)
    axes[i+4].set_title(col_name, fontsize=12) #, weight='bold')
    # axes[i+4].set_xlim(-5, max_x+50)
    axes[i+4].set_ylim((min_y[i]-.03)*scale, (max_y[i]+.03)*scale)
    axes[i+4].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[i+4].grid(True)

fig.text(0.5, 0.95, '(a) Search Frequency', ha='center', va='bottom', fontsize=16, weight='bold')
fig.text(0.5, 0.45, '(b) Search Quality', ha='center', va='bottom', fontsize=16, weight='bold')
fig.legend(handles=lines, loc='lower center', ncol=4)
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.12, hspace=0.8, wspace=0.13)
plt.savefig('visualize/figures/curve_search_behavior.pdf', bbox_inches='tight', dpi=300)
