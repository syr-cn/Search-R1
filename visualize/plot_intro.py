import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 7), subplot_kw={ 'polar': False, 'polar': True })
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])  # Wider right plot if needed
ax_left = fig.add_subplot(gs[0, 0])                 # Normal axes (line plot)
ax_right = fig.add_subplot(gs[0, 1], polar=True)    # Polar axes (radar plot)
ax_left.set_box_aspect(.7)

# === Right Panel: Line Plot ===
# line_palette = sns.color_palette("Set2", n_colors=5)
line_palette = sns.color_palette("husl", n_colors=3)

csv_names = [
    'visualize/wandb/ours-base-f1-wScore.csv',
    'visualize/wandb/searchr1-base-may8.csv',
    'visualize/wandb/research-base-may9.csv',
]
line_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
col_name = 'env/number_of_valid_search'
# col_name = 'val/number_of_valid_search/mean'
col_name = 'val/number_of_valid_search/'
exp_names = [
    'AutoRefine',
    'Search-R1',
    'ReSearch',
    'x=1',
]

ds_keys = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
ds_values = ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2Wiki", "Musique", "Bamboogle"]

# Prepare data
bar_values = []
for df in line_dfs:
    bar_values.append([df[col_name+label][100] for label in ds_keys])

# Convert to numpy array for easier indexing: shape (3, 7)
bar_values = np.array(bar_values)
# Bar plot settings
y = np.arange(len(ds_keys))  # vertical positions
height = 0.25  # thickness of bars

# Clear previous plot
ax_left.clear()

# Plot horizontal bars
ax_left.barh(len(ds_keys)-1-(y - height), bar_values[0], height, label=exp_names[0], color=line_palette[0], zorder=6, alpha=0.8)
ax_left.barh(len(ds_keys)-1-(y ),         bar_values[1], height, label=exp_names[1], color=line_palette[1], zorder=5, alpha=0.8)
ax_left.barh(len(ds_keys)-1-(y + height), bar_values[2], height, label=exp_names[2], color=line_palette[2], zorder=4, alpha=0.8)

# Add vertical line at x = 1.0 to distinguish single vs multi-hop
ax_left.axvline(x=1.0, color='orange', linewidth=1.5, linestyle='--', label=exp_names[-1], zorder=3)

# Axes formatting
ax_left.set_xlabel("# Search Calls per Sample", fontsize=12)
ax_left.set_yticks(y)
ax_left.set_yticklabels(ds_values[::-1], fontsize=10)
ax_left.set_xlim(0.55, 2.05)
ax_left.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
# ax_left.grid(True, axis='x', linestyle='--', alpha=0.5)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# Tweak layout
ax_left.set_title("(a)", x=-0.2, y=1.02, fontsize=18, weight='bold')
ax_left.legend(loc='upper right', ncol=1, fontsize=10)


# === Right Panel: Radar Plot ===
radar_csv = "visualize/data/3b_results_allowed.csv"
df_radar = pd.read_csv(radar_csv)

radar_palette = sns.color_palette("husl", n_colors=len(df_radar))
methods = df_radar["Methods"]
categories = df_radar.columns[1:]
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Normalize
max_vals = df_radar[categories].max()
outer_limits = np.floor((max_vals + 0.05) * 20) / 20
norm_df = df_radar.copy()
for col in categories:
    norm_df[col] = df_radar[col] / outer_limits[col]
outer_box = [1] * num_vars + [1]
ax_right.plot(angles, outer_box, color='gray', linewidth=1.5, linestyle='-')

for angle in angles[:-1]:
    ax_right.plot([angle, angle], [0, 1], color='gray', linewidth=0.8)

for i, category in enumerate(categories):
    angle_rad = angles[i]
    ha = "left" if np.cos(angle_rad) > 0 else "right"
    va = "bottom" if np.sin(angle_rad) > 0 else "top"
    ax_right.text(angle_rad, 1.02, f"{outer_limits[category]:.2f}", size=8,
            horizontalalignment=ha, verticalalignment=va, color="gray")

for i in range(len(norm_df)):
    values = norm_df.iloc[i, 1:].tolist() + [norm_df.iloc[i, 1]]
    ax_right.plot(angles, values, linewidth=2, label=methods[i], color=radar_palette[i])
    ax_right.fill(angles, values, alpha=0.07, color=radar_palette[i])

ax_right.set_frame_on(False)
ax_right.xaxis.set_visible(False)
ax_right.yaxis.set_visible(False)

for i, category in enumerate(categories):
    angle_rad = angles[i]
    ha = "left" if np.cos(angle_rad) > 0 else "right"
    va = "bottom" if np.sin(angle_rad) > 0 else "top"
    ax_right.text(angle_rad, 1.2, category, size=12,
            horizontalalignment=ha, verticalalignment=va, weight='bold')

ax_right.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=10)
ax_right.set_title("(b)", x=-0.3, y=1.0, fontsize=18, weight='bold')

# plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.8, wspace=0.4)
plt.savefig("visualize/figures/intro_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
