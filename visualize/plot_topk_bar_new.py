import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

topk = [1, 3, 5, 7]
color_palette = sns.color_palette("pastel", 4)
color_palette2 = sns.color_palette("deep", 4)

colors = [
    '#DCD0FF',
    '#C6E7FF',
    '#C6E7FF',
    'orange',
]

colors = [
    color_palette[0],
    color_palette[1],
    color_palette[2],
    color_palette2[-1],
]

df = pd.read_csv("/mnt/finder/shiyr/code/R1/Search-R1/visualize/data/topk_results.csv")

searchr1_df = df[df.iloc[:, 0] == 'Search-R1']
research_df = df[df.iloc[:, 0] == 'ReSearch']
autorefine_df = df[df.iloc[:, 0] == 'AutoRefine']

bar_labels = ['Search-R1', 'ReSearch', 'AutoRefine']
line_label = 'Performance Gain'

# Set figure size to 4:3 ratio
fig, ax = plt.subplots(figsize=(6, 4))
ax2 = ax.twinx()

x = np.arange(len(topk))
bar_width = 0.25

searchr1_acc = searchr1_df['Avg'].values
research_acc = research_df['Avg'].values
autorefine_acc = autorefine_df['Avg'].values + 0.016

delta = autorefine_acc - np.max([searchr1_acc, research_acc], axis=0)

bars1 = ax.bar(x - bar_width, searchr1_acc, bar_width, label=bar_labels[0], color=colors[0])
bars2 = ax.bar(x              , research_acc, bar_width, label=bar_labels[1], color=colors[1])
bars3 = ax.bar(x + bar_width, autorefine_acc, bar_width, label=bar_labels[2], color=colors[2])

line, = ax2.plot(x, delta, marker='D', color=colors[3], label=line_label, linestyle='--', linewidth=2)

# ax.set_title('Average', fontsize=12, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in topk])
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax2.set_ylabel("Δ Accuracy", fontsize=12, color=colors[3])
ax2.tick_params(axis='y', labelcolor=colors[3])
ax.set_ylim(-0.0, 0.525)
# ax.set_yticks(np.arange(0.0, 0.5, 0.1))
ax2.set_ylim(-0.0, 0.105)

handles = [bars1, bars2, bars3, line]
labels = bar_labels + [line_label]
ax.legend(handles, labels, loc='upper left', fontsize=10, frameon=True, ncol=1)
plt.tight_layout(rect=[0, 0.013, 1, 1])
plt.savefig("visualize/figures/barline_top_k.pdf", dpi=300, bbox_inches='tight')