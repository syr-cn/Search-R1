import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

topk = [1, 3, 5, 7]

df = pd.read_csv("/mnt/finder/shiyr/code/R1/Search-R1/visualize/data/topk_results.csv")

search_base_df = df[df.iloc[:, 0] == 'Search-R1-base']
ours_df = df[df.iloc[:, 0] == 'Ours-base']

bar_labels = ['Search-R1-Base', 'AutoRefine-Base']
line_label = 'Performance Gain'

# Set figure size to 4:3 ratio
fig, ax = plt.subplots(figsize=(4, 3))
ax2 = ax.twinx()

x = np.arange(len(topk))
bar_width = 0.35

base_acc = search_base_df['Avg'].values
ours_acc = ours_df['Avg'].values
delta = ours_acc - base_acc

bars1 = ax.bar(x - bar_width/2, base_acc, bar_width, label=bar_labels[0], color='#DCD0FF')
bars2 = ax.bar(x + bar_width/2, ours_acc, bar_width, label=bar_labels[1], color='#C6E7FF')

line, = ax2.plot(x, delta, marker='o', color='orange', label=line_label, linestyle='--', linewidth=2)

ax.set_title('Average', fontsize=12, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in topk])
ax.set_xlabel("#Doc per Search", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
ax2.set_ylabel("Δ Accuracy", fontsize=10, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Legend
# handles = [bars1, bars2, line]
# labels = [bar_labels[0], bar_labels[1], line_label]
# fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

# plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.savefig("barline_avg_only.png", dpi=300, bbox_inches='tight')
handles = [bars1, bars2, line]
labels = [bar_labels[0], bar_labels[1], line_label]
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("barline_avg_only.png", dpi=300, bbox_inches='tight')