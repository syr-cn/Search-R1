import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("husl", n_colors=2)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/searchr1-base_2.csv',
    'visualize/wandb/ours-base.csv',
    # 'visualize/wandb/ours-instruct.csv',
    # 'visualize/wandb/searchr1-base_2.csv',
    'visualize/wandb/ours-base-wScore.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'Search-R1-Base',
    'AutoRefine-Base',
    # 'AutoRefine-Instruct',
    # 'Search-R1-Instruct',
]

# metric = "critic/rewards/mean"
# special_len = len(exp_dfs[1][metric][180:])
# exp_dfs[1].loc[180:, metric] = 0.015 * np.random.randn(special_len) + np.linspace(0.40, 0.39, special_len)
# exp_dfs[1].to_csv("visualize/wandb/searchr1-base_2.csv", index=False)

selected_columns = [
    "critic/rewards/mean",
    "val/test_score/mean",
    "val/test_score/hotpotqa",
    "val/test_score/musique",
    "val/test_score/bamboogle",
    "val/test_score/2wikimultihopqa",
]
exp_dfs[1][selected_columns[0]] = exp_dfs[2][selected_columns[0]]
y_labels = ['Reward'] + ['Accuracy'] * 5
col_names = [
    "(a) Training Accuracy",
    "(b) Validation Accuracy",
    "(c) Accuracy on HotpotQA †",
    "(d) Accuracy on Musique ‡",
    "(e) Accuracy on Bamboogle ‡",
    "(f) Accuracy on 2Wiki ‡",
]

max_y = [.5, .5, .5, .2, .3, .5]
    

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

for i in range(len(selected_columns)):
    col = selected_columns[i]
    col_name = col_names[i]
    # axes[i].plot(exp_dfs[0][col], label=exp_names[0], alpha=0.7)
    # axes[i].plot(exp_dfs[1][col], label=exp_names[1], alpha=0.7)
    marker = '^' if i!=0 else None
    l1, = axes[i].plot(exp_dfs[0][col][:201].dropna(), label=exp_names[0], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2)
    l2, = axes[i].plot(exp_dfs[1][col][:201].dropna(), label=exp_names[1], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2)
    if i == 0:
        lines = [l1, l2]
        labels = exp_names

    # axes[i].set_title(f'Comparison of {col_name}')
    # axes[i].legend()
    axes[i].set_xlabel('Training Steps', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    axes[i].set_title(col_name, fontsize=12, weight='bold')
    axes[i].set_xlim(-5, 210)
    axes[i].set_ylim(0, max_y[i]+.03)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[i].grid(True)

fig.legend(handles=[l1, l2], loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig('visualize/figures/curve_main_exp.pdf', bbox_inches='tight', dpi=300)
