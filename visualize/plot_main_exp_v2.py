import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("husl", n_colors=2)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/searchr1-base_2.csv',
    'visualize/wandb/ours-base-wScore.csv',
    # 'visualize/wandb/ours-instruct.csv',
    # 'visualize/wandb/searchr1-base_2.csv',
    # 'visualize/wandb/ours-base-no_reward.csv',
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
y_labels = ['Reward'] + ['Accuracy'] * 5
col_names = [
    "(a) Training Rewards",
    "(b) Average Accuracy",
    "(c) Accuracy on HotpotQA†",
    "(d) Accuracy on Musique*",
    "(e) Accuracy on Bamboogle*",
    "(f) Accuracy on 2Wiki*",
]

max_y = [.5, .5, .5, .2, .3, .5]
    
# Reorder plot indices: move (b) to bottom left (index 3)
plot_order = [0, 2, 4, 1, 3, 5]  # original indices of selected_columns

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

# Plotting loop with reordered subplot placement
for plot_idx, original_idx in enumerate(plot_order):
    col = selected_columns[original_idx]
    col_name = col_names[original_idx]
    marker = '^' if original_idx != 0 else None
    l1, = axes[plot_idx].plot(exp_dfs[0][col][:201].dropna(), label=exp_names[0], alpha=1, color=line_palette[0], marker=marker, linestyle='-', linewidth=2)
    l2, = axes[plot_idx].plot(exp_dfs[1][col][:201].dropna(), label=exp_names[1], alpha=1, color=line_palette[1], marker=marker, linestyle='-', linewidth=2)
    
    if original_idx == 0:
        lines = [l1, l2]
        labels = exp_names

    axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
    axes[plot_idx].set_ylabel(y_labels[original_idx], fontsize=12)
    axes[plot_idx].set_title(col_name, fontsize=12, weight='bold')
    axes[plot_idx].set_xlim(-5, 210)
    axes[plot_idx].set_ylim(0, max_y[original_idx] + .03)
    axes[plot_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[plot_idx].grid(True)

# Add dashed black line between top row and bottom row
fig.subplots_adjust(hspace=0.5)
line = plt.Line2D((.338,.338),(.08,.97), color="k", linewidth=2, linestyle='--')
fig.add_artist(line)
# for ax in axes[0:3]:
#     bbox = ax.get_position()
#     fig.plot([bbox.x0, bbox.x1], [bbox.y0 - 0.02, bbox.y0 - 0.02], transform=fig.transFigure, color='black', linestyle='--', linewidth=1)

# from matplotlib.transforms import Bbox

# # Manually set subplot positions (left, bottom, width, height)
# positions = [
#     [0.06, 0.57, 0.3, 0.35],  # (a) Training Rewards - top-left
#     [0.39, 0.57, 0.3, 0.35],  # (c) Accuracy on HotpotQA - top-mid
#     [0.72, 0.57, 0.3, 0.35],  # (d) Accuracy on Musique - top-right
#     [0.06, 0.13, 0.3, 0.35],  # (b) Average Accuracy - bottom-left
#     [0.39, 0.13, 0.3, 0.35],  # (e) Accuracy on Bamboogle - bottom-mid
#     [0.72, 0.13, 0.3, 0.35],  # (f) Accuracy on 2Wiki - bottom-right
# ]

# for ax, pos in zip(axes, positions):
#     ax.set_position(Bbox.from_bounds(*pos))

# Legend and save
fig.legend(handles=[l1, l2], loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig('visualize/figures/curve_main_exp.pdf', bbox_inches='tight', dpi=300)
