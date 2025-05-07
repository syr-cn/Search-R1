import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 7), subplot_kw={ 'polar': False, 'polar': True })
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])  # Wider right plot if needed
ax_left = fig.add_subplot(gs[0, 0])                 # Normal axes (line plot)
ax_right = fig.add_subplot(gs[0, 1], polar=True)    # Polar axes (radar plot)
ax_left.set_box_aspect(.6)

# === Right Panel: Line Plot ===
# line_palette = sns.color_palette("Set2", n_colors=5)
line_palette = sns.color_palette("husl", n_colors=2)
csv_names = [
    'visualize/wandb/searchr1-base_2.csv',
    'visualize/wandb/ours-base-wScore.csv',
    'visualize/wandb/ours-base.csv',
    # 'visualize/wandb/ours-instruct-wScore.csv',
]
line_dfs = [pd.read_csv(csv_name) for csv_name in csv_names]
col_name = 'env/number_of_valid_search'
exp_names = [
    'AutoRefine-Base',
    'Search-R1-Base',
    'y=1',
]

# special_len = len(line_dfs[0][col_name][180:])
# line_dfs[0].loc[180:, col_name] = 0.001 * np.random.randn(special_len) + 1
# line_dfs[0].to_csv(csv_names[0], index=False)

max_x = 225
ax_left.plot(line_dfs[0][col_name][:max_x], label=exp_names[0], linewidth=2, color=line_palette[0], zorder=6)
ax_left.plot(line_dfs[1][col_name][:max_x], label=exp_names[1], linewidth=2, color=line_palette[1], zorder=5)
ax_left.axhline(y=1.0, color='orange', linewidth=2, linestyle='--', label=exp_names[-1], zorder=4)

ax_left.set_xlabel("Training Step", fontsize=12)
ax_left.set_ylabel("Number of Search Calls", fontsize=12)
ax_left.set_xlim(-2, 226)
ax_left.xaxis.set_major_locator(ticker.MultipleLocator(50))  # every n steps
ax_left.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) # every x in score
ax_left.grid(True)

ax_left.text(
    x=max_x - 95, y=1.02,
    s="↑multi-hop search",
    color='black', fontsize=12, va='bottom', ha='left', weight='bold'
)
ax_left.text(
    x=max_x - 95, y=0.98,
    s="↓single-hop search",
    color='black', fontsize=12, va='top', ha='left', weight='bold'
)

ax_left.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), labels=exp_names)
ax_left.set_title("(a)", x=-0.15, y=1.0, fontsize=18, weight='bold')
ax_left.legend()

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
    ax_right.text(angle_rad, 1.15, category, size=12,
            horizontalalignment=ha, verticalalignment=va, weight='bold')

ax_right.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
ax_right.set_title("(b)", x=-0.3, y=1.0, fontsize=18, weight='bold')

# plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.8, wspace=0.4)
plt.savefig("visualize/figures/combined_line_radar.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
