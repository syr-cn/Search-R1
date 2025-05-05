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
datasets = ['Avg', 'NQ', 'HotpotQA', 'Musique']
search_base = {key: search_base_df[key].tolist() for key in datasets}
ours = {key: ours_df[key].tolist() for key in datasets}

titles = ['(a) Avg', '(b) NQ', '(c) HotpotQA', '(d) Musique']
metrics = ['Avg', 'NQ', 'HotpotQA', 'Musique']
y_lims = [0.45, 0.48, 0.42, 0.18]  
line_palette = sns.color_palette("Set2", n_colors=5)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(topk, ours[metric], marker='o', label='Ours',
            color=line_palette[0], linewidth=2)
    ax.plot(topk, search_base[metric], marker='s', label='Search-R1-Base',
            color=line_palette[1], linewidth=2)

    ax.set_title(titles[i], fontsize=12)
    ax.set_xlabel('Top-K', fontsize=11)
    if i == 0:
        ax.set_ylabel('EM', fontsize=11)

    ax.set_ylim(0, y_lims[i])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# plt.grid()
fig.legend(['Ours', 'Search-R1-Base'], loc='lower center', ncol=2, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

plt.savefig('visualize/figures/curve_top_k.pdf', dpi=300, bbox_inches='tight')
