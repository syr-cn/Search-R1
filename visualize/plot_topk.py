import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

topk = [1, 3, 5, 7]
# Load the CSV file
df = pd.read_csv("visualize/data/topk_results.csv")

# Filter by model
search_base_df = df[df.iloc[:, 0] == 'Search-R1-base']
ours_df = df[df.iloc[:, 0] == 'Ours-base']

# Extract required metrics into dictionaries
# ds_names = ['Avg', 'NQ', 'HotpotQA', 'Musique']
ds_names = ['Avg', 'HotpotQA', 'Bamboogle', 'Musique']
search_base = {key: search_base_df[key].tolist() for key in ds_names}
ours = {key: ours_df[key].tolist() for key in ds_names}

# titles = ['(a) Average', '(b) NQ', '(c) HotpotQA', '(d) Musique']
titles = ['(a) Average', '(b) HotpotQA', '(c) Bamboogle', '(d) Musique']
labels = ['Search-R1-Base', 'AutoReThink-Base', 'Performance Gain']
y_lims = [
    (0, 0.45),
    (0, 0.45),
    (0, 0.45),
    (0, 0.17),
]
line_palette = sns.color_palette("husl", n_colors=2)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True)
axes = axes.flatten()

for i, ds_name in enumerate(ds_names):
    ax = axes[i]
    ax.plot(topk, search_base[ds_name], marker='^', label=labels[0], color=line_palette[0], linewidth=2)
    ax.plot(topk, ours[ds_name], marker='^', label=labels[1], color=line_palette[1], linewidth=2)
    ax.plot(topk, [i-j for i, j in zip(ours[ds_name], search_base[ds_name])], marker='s', label=labels[2], color='orange', linestyle='--', linewidth=2)

    ax.set_title(titles[i], fontsize=12, weight='bold')
    ax.set_xticks(list(range(max(topk)+1)))
    ax.set_xlim(.7, max(topk)+.3)
    ax.set_xlabel('#Doc per Search', fontsize=12)
    if i == 0:
        ax.set_ylabel('Accuracy', fontsize=11)

    ax.set_ylim(*y_lims[i])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True)

# plt.grid()
fig.legend(labels, loc='lower center', ncol=3, fontsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.24)

plt.savefig('visualize/figures/curve_top_k.pdf', dpi=300, bbox_inches='tight')
