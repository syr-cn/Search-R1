import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import numpy as np

# line_palette = sns.color_palette("husl", n_colors=2)
line_palette2 = sns.color_palette("Set2", n_colors=4)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-base-wScore.csv',
    'visualize/wandb/ours-base.csv',
    # 'visualize/wandb/ours-instruct.csv',
    # 'visualize/wandb/searchr1-base_2.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]

selected_columns = [
    "critic/info_score/mean",
    "critic/refine_score/mean",
    "critic/answer_scores/mean",
]
    
max_x = 200
y_lim = [
    (0, 73),
    (0, 73),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes = axes.flatten()
exp_names = [
    'Ground Truth in Documents',
    'Ground Truth in Refinement',
    'Answer Accuracy',
]

axes[0].plot(exp_dfs[0][selected_columns[0]][:max_x].dropna()*100, label=exp_names[0], alpha=1, color=line_palette2[0], linestyle='-', linewidth=2, zorder=6)
axes[0].plot(exp_dfs[0][selected_columns[1]][:max_x].dropna()*100, label=exp_names[1], alpha=1, color=line_palette2[1], linestyle='-', linewidth=2, zorder=5)
axes[0].plot(exp_dfs[0][selected_columns[2]][:max_x].dropna()*100, label=exp_names[2], alpha=1, color=line_palette2[2], linestyle='-', linewidth=2, zorder=4)
axes[0].legend(loc='lower right', fontsize=10)


exp_names = [
    'GT in Ref w/ Retrieval Reward',
    'GT in Ref w/o Retrieval Reward',
]

axes[1].plot(exp_dfs[0][selected_columns[1]][:max_x].dropna()*100, label=exp_names[0], alpha=1, color=line_palette2[1], linestyle='-', linewidth=2, zorder=6)
axes[1].plot(exp_dfs[1][selected_columns[1]][:max_x].dropna()*100, label=exp_names[1], alpha=1, color=line_palette2[3], linestyle='-', linewidth=2, zorder=5)
axes[1].legend(loc='lower right', fontsize=10)


for i in range(2):
    axes[i].set_xlabel('Training Steps', fontsize=12)
    # if y_labels[i]:
    #     axes[i].set_ylabel(y_labels[i], fontsize=12)
    # axes[i].set_title(col_names[i], fontsize=12, weight='bold')
    axes[i].set_xlim(-5, max_x+5)
    axes[i].set_ylim(*y_lim[i])
    
    axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
    # axes[i].set_yticks(y_ticks[i])
    axes[i].grid(True)

plt.tight_layout()
# plt.subplots_adjust(bottom=0.23, top=0.9)
plt.savefig('visualize/figures/curve_refine_quality.pdf', bbox_inches='tight', dpi=300)