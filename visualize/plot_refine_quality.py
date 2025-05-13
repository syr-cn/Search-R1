import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import numpy as np

# line_palette = sns.color_palette("husl", n_colors=2)
color_palette = sns.color_palette("viridis_r")
color_palette2 = sns.color_palette("Set2", n_colors=6)

colors = [
    color_palette[0],
    color_palette[2],
    color_palette[4],
]

# Step 1: Read the two CSV files
csv_names = [
    'visualize/wandb/ours-f1-refScore.csv',
]
exp_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]

selected_columns = [
    "critic/info_score/mean",
    "critic/refine_score/mean",
    "critic/answer_scores/mean",
]

max_x = 172
y_lim = [
    (0, 83),
    (0, 83),
]
fig, axes = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[1.6, 1])
axes = axes.flatten()
exp_names = [
    'Search',
    'Refine',
    'Answer',
]

axes[0].plot(exp_dfs[0][selected_columns[0]][:max_x].dropna()*100, label=exp_names[0], alpha=1, color=colors[0], linestyle='-', linewidth=2, zorder=6)
axes[0].plot(exp_dfs[0][selected_columns[1]][:max_x].dropna()*100, label=exp_names[1], alpha=1, color=colors[1], linestyle='-', linewidth=2, zorder=5)
axes[0].plot(exp_dfs[0][selected_columns[2]][:max_x].dropna()*100, label=exp_names[2], alpha=1, color=colors[2], linestyle='-', linewidth=2, zorder=4)
axes[0].legend(loc='lower right', fontsize=10)

i=0
axes[i].set_xlabel('Training Steps', fontsize=12)
# if y_labels[i]:
#     axes[i].set_ylabel(y_labels[i], fontsize=12)
# axes[i].set_title(col_names[i], fontsize=12, weight='bold')
axes[i].set_xlim(-5, max_x+2)
axes[i].set_xticks(np.arange(0, max_x+1, 25))
axes[i].set_ylim(*y_lim[i])

# axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
# axes[i].set_yticks(y_ticks[i])
axes[i].grid(True)
axes[0].set_ylabel('(a) Success Rate (%)', fontsize=12, weight='bold')


val_path = 'log/val/nq_hotpotqa_train_autorefine-grpo-qwen2.5-3b-f1-ref0.1.jsonl'

len_lists = {
    'documents': [],
    'refine': [],
    'answer': [],
}

import re

def extract_answer(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return ''
    return [matches[-1].group(1).strip()]

def extract_documents(solution_str):
    doc_pattern = r'<documents>(.*?)<documents>'
    match = re.finditer(doc_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return ''
    return list([m.group(1).strip() for m in matches])

def extract_refine(solution_str):
    doc_pattern = r'<refine>(.*?)\n'
    match = re.finditer(doc_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return ''
    return list([m.group(1).strip() for m in matches])


import tiktoken
enc = tiktoken.get_encoding("o200k_base")
def token_count(text_list):
    if len(text_list) == 0:
        return 0
    return sum([len(enc.encode(text)) for text in text_list])#/len(text_list)

import json
with open(val_path, 'r') as f:
    result_list = [json.loads(line) for line in f.readlines() if len(line) > 0]
    result_list = result_list[-1000:]

for data_item in result_list:
    solution_str = data_item['response']
    answer = extract_answer(solution_str)
    refine = extract_refine(solution_str)
    documents = extract_documents(solution_str)
    len_lists['answer'].append(token_count(answer))
    len_lists['refine'].append(token_count(refine))
    len_lists['documents'].append(token_count(documents))

for key in len_lists:
    zero_idxs = [np.where(np.array(len_lists[key]) == 0)[0]]
    print(f"key: {key}, zero ratio: {len(zero_idxs)/len(len_lists[key])*100:.3f}%")
    len_lists[key] = [i for i in len_lists[key] if i > 0]

data = [
    len_lists['documents'],
    len_lists['refine'],
    len_lists['answer'],
]
name_list = [
    'Documents',
    'Refinement',
    'Answer',
]
parts = axes[1].violinplot(data, showmeans=True, showextrema=True)

# Apply custom colors to each violin
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.3)

for i, color in enumerate(colors):
    for line_type in ['cmins', 'cmaxes']:
        line = parts[line_type]
        line.get_segments()[i][:]
        line.set_color(colors)
    parts['cbars'].set_color(colors)
    parts['cmeans'].set_color(colors)

# box = axes[1].boxplot(
#     data,
#     widths=0.15,
#     patch_artist=True,
#     boxprops=dict(facecolor='none', linewidth=.8),
#     capprops=dict(linewidth=.8),
#     whiskerprops=dict(linewidth=.8),
#     flierprops=dict(marker='.', markersize=3, linestyle='none'),
#     medianprops=dict(linewidth=.8),
#     positions=[1, 2, 3]
# )

# # Apply color[i] to each box component
# for i in range(len(colors)):
#     color = colors[i]
#     box['boxes'][i].set_edgecolor(color)
#     box['caps'][2*i].set_color(color)
#     box['caps'][2*i+1].set_color(color)
#     box['whiskers'][2*i].set_color(color)
#     box['whiskers'][2*i+1].set_color(color)
#     box['medians'][i].set_color(color)
#     box['fliers'][i].set_markerfacecolor(color)
#     box['fliers'][i].set_markeredgecolor(color)

# Set x-axis labels and y-axis to log scale
axes[1].set_xticks(range(1, len(name_list) + 1))
axes[1].set_xticklabels(name_list)
axes[1].set_xlabel('Components', fontsize=12)
axes[1].grid(True, linestyle='-', alpha=0.5, zorder=0)
axes[1].set_ylabel('(b) Number of Tokens', fontsize=12, weight='bold')

plt.tight_layout()
# plt.subplots_adjust(bottom=0.23, top=0.9)
plt.savefig('visualize/figures/curve_refine_quality.pdf', bbox_inches='tight', dpi=300)