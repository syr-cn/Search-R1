import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

topk = [1, 3, 5, 7]

df = pd.read_csv("/mnt/finder/shiyr/code/R1/Search-R1/visualize/data/topk_results.csv")

search_base_df = df[df.iloc[:, 0] == 'Search-R1-base']
ours_df = df[df.iloc[:, 0] == 'Ours-base']

ds_names = ['Avg', 'HotpotQA', 'Bamboogle', 'Musique']
titles = ['(a) Average', '(b) HotpotQA', '(c) Bamboogle', '(d) Musique']
bar_labels = ['Search-R1-Base', 'AutoRefine-Base']
line_label = 'Performance Gain'

fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
axes = axes.flatten()

bar_width = 0.35
x = np.arange(len(topk))
y_lims = [.43,.43,.43,.18,]
y_lims2 = [.11,.11,.26,.11,]


for i, ds_name in enumerate(ds_names):
    ax = axes[i]
    ax2 = ax.twinx()  

    base_acc = search_base_df[ds_name].values
    ours_acc = ours_df[ds_name].values
    delta = ours_acc - base_acc

    bars1 = ax.bar(x - bar_width/2, base_acc, bar_width, label=bar_labels[0], color='#DCD0FF')
    bars2 = ax.bar(x + bar_width/2, ours_acc, bar_width, label=bar_labels[1], color='#C6E7FF')

    line, = ax2.plot(x, delta, marker='o', color='orange', label=line_label,linestyle='--', linewidth=2)

    ax.set_title(titles[i], fontsize=12, weight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in topk])
    ax.set_xlabel("#Doc per Search", fontsize=10)

    if i == 0:
        ax.set_ylabel("Accuracy", fontsize=10)
    if i == len(ds_names) - 1:
        ax2.set_ylabel("Δ Accuracy", fontsize=10, color='orange')

    ax2.tick_params(axis='y', labelcolor='orange')
    ax.set_ylim(0, y_lims[i])
    ax2.set_ylim(0, y_lims2[i])

handles = [bars1, bars2, line]
labels = [bar_labels[0], bar_labels[1], line_label]
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.1, 1, 1], pad=.2)
plt.savefig("visualize/figures/barline_top_k_combined.pdf", dpi=300, bbox_inches='tight')