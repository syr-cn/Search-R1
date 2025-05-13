import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

line_palette = sns.color_palette("rocket", n_colors=3)

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-f1-refScore-2.csv',
    'visualize/wandb/ours-base.csv',
    'visualize/wandb/searchr1-base_2.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]

exp_names = [
    'w/ Retrieval Reward',
    'w/o Retrieval Reward',
    'w/o Retrieval Reward & Refinement',
]
exp_dfs[1]['critic/answer_scores/mean'] = exp_dfs[1]['critic/rewards/mean']
exp_dfs[2]['critic/answer_scores/mean'] = exp_dfs[2]['critic/rewards/mean']

selected_columns = [
    # 'val/test_score/mean',
    # 'critic/answer_scores/mean',
    "env/number_of_valid_search",
    'critic/info_score/mean',
    'critic/refine_score/mean',
]

titles = [
    '(a) Search Frequency',
    '(b) Search Quality',
    '(c) Refinement Quality',
]


y_labels = [
    '# Search Calls',
    'Success Rate (%)',
    'Success Rate (%)',
]
    
max_x = 165
y_lim = [
    (0.58, 2.55),
    (13, 82),
    (5, 82),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes = axes.flatten()

# idx = 0
for idx in range(3):
    scale = 1 if idx==0 else 100
    l1, = axes[idx].plot(exp_dfs[0][selected_columns[idx]][:max_x].dropna()*scale, label=exp_names[0], alpha=1, color=line_palette[0], linestyle='-', linewidth=1.5, zorder=6)
    l2, = axes[idx].plot(exp_dfs[1][selected_columns[idx]][:max_x].dropna()*scale, label=exp_names[1], alpha=1, color=line_palette[1], linestyle='-', linewidth=1.5, zorder=5)
    l3, = axes[idx].plot(exp_dfs[2][selected_columns[idx]][:max_x].dropna()*scale, label=exp_names[2], alpha=1, color=line_palette[2], linestyle='-', linewidth=1.5, zorder=4)
    if idx == 0:
        lines = [l1, l2, l3]
    # axes[idx].legend(loc='upper left', fontsize=10, frameon=False)

for i in range(3):
    axes[i].set_xlabel('Training Steps', fontsize=10)
    if y_labels[i]:
        axes[i].set_ylabel(y_labels[i], fontsize=12)
    # axes[i].set_title(y_titles[i], fontsize=12, weight='bold')
    # axes[i].set_xlim(-5, max_x+5)
    axes[i].set_ylim(*y_lim[i])
    if titles[i]:
        axes[i].set_title(titles[i], fontsize=12, weight='bold')
    axes[i].grid(True)

fig.legend(handles=lines, loc='lower center', ncol=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.26, top=0.9)
plt.savefig('visualize/figures/curve_ablation.pdf', bbox_inches='tight', dpi=300)