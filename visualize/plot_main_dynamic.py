import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("Set2", n_colors=4)

colors = [
    line_palette[0],
    line_palette[3],
]

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-f1-refScore-2.csv',
    'visualize/wandb/ours-instruct-f1-refScore.csv',
    'visualize/wandb/ours-instruct.csv',
    'visualize/wandb/searchr1-base-may8.csv',
    'visualize/wandb/research-base-may9.csv',
]

exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
exp_names = [
    'AutoRefine-Base',
    'AutoRefine-Instruct',
    'Search-R1-Base',
    'ReSearch-Base',
]

# metric = "critic/rewards/mean"
# special_len = len(exp_dfs[1][metric][180:])
# exp_dfs[1].loc[180:, metric] = 0.015 * np.random.randn(special_len) + np.linspace(0.40, 0.39, special_len)
# exp_dfs[1].to_csv("visualize/wandb/searchr1-base_2.csv", index=False)

selected_columns = [
    "critic/rewards/mean",
    "val/test_score/mean",
    "response_length/mean",
    "env/number_of_valid_search",
]
# for col in selected_columns:
#     exp_dfs[1][col] = exp_dfs[1][col][:122]

y_labels = [
    "Training Reward",
    # "# Search Calls",
    "Validation Accuracy",
    "Response length",
]

y_lims = [
    (.07, .73),
    (.07, .53),
    # (.87, 2.03),
    (600, 1600),
]
x_max=181
    
# Reorder plot indices: move (b) to bottom left (index 3)
plot_order = [0, 1, 2]  # original indices of selected_columns

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
axes = axes.flatten()

# Plotting loop with reordered subplot placement
for plot_idx, original_idx in enumerate(plot_order):
    col = selected_columns[original_idx]
    # col_name = col_names[original_idx]
    marker = '^' if original_idx >10 else None
    l1, = axes[plot_idx].plot(exp_dfs[0][col][:x_max].dropna(), label=exp_names[0], alpha=1, color=colors[0], marker=marker, linestyle='-', linewidth=2, zorder=5)
    l2, = axes[plot_idx].plot(exp_dfs[1][col][:x_max].dropna(), label=exp_names[1], alpha=1, color=colors[1], marker=marker, linestyle='-', linewidth=2, zorder=4)
    
    if original_idx == 0:
        lines = [l1, l2]
        labels = exp_names

    axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
    axes[plot_idx].set_ylabel(y_labels[original_idx], fontsize=12)
    # axes[plot_idx].set_title(col_name, fontsize=12, weight='bold')
    axes[plot_idx].set_xlim(-5, x_max+2)
    axes[plot_idx].set_ylim(*y_lims[original_idx])
    # axes[plot_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[plot_idx].grid(True)


# Legend and save
fig.legend(handles=lines, loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.26)
plt.savefig('visualize/figures/curve_main_dynamic.pdf', bbox_inches='tight', dpi=300)
