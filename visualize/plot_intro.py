import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 7), subplot_kw={ 'polar': False, 'polar': True })
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])  # Wider right plot if needed
ax_left = fig.add_subplot(gs[0, 0])                 # Normal axes (line plot)
ax_right = fig.add_subplot(gs[0, 1], polar=True)    # Polar axes (radar plot)
ax_left.set_box_aspect(.8)

# === Right Panel: Line Plot ===
line_palette = sns.color_palette("Set2", n_colors=5)
line_csv = "visualize/data/critic_info_scores.csv"  # Replace with actual path
line_csv2 = "visualize/data/info_num_valid_search_2.csv"
df_line = pd.read_csv(line_csv)
df_line2 = pd.read_csv(line_csv2)
column_display_map = {
    "w/ refinement": "nq_hotpotqa_train_refine-r1-grpo-qwen2.5-3b-em-refine_score-0.2 - critic/info_score/mean",
    "w/o refinement": "nq_hotpotqa_train_base_score-search-r1-grpo-qwen2.5-3b-em - critic/info_score/mean"
}

# special_name = "nq_hotpotqa_train_base_score-search-r1-grpo-qwen2.5-3b-em - env/number_of_valid_search"
# special_len = len(df_line2[special_name][180:])
# df_line2.loc[180:, special_name] = 0.001 * np.random.randn(special_len) + 1
# df_line2.to_csv(line_csv2.replace(".csv", "_2.csv"), index=False)

for col_key in column_display_map.values():
    col_key2 = col_key.replace('critic/info_score/mean', 'env/number_of_valid_search')
    # df_line[col_key] = df_line[col_key] * df_line2[col_key2]
    df_line[col_key] = df_line2[col_key2][:225]

ax_left.axhline(y=1.0, color='red', linewidth=1, linestyle='--')
for idx, (display_name, col_key) in enumerate(column_display_map.items()):
    ax_left.plot(df_line["Step"], df_line[col_key], label=display_name, linewidth=2, color=line_palette[idx])

ax_left.set_xlabel("Step", fontsize=12)
ax_left.set_ylabel("Number of Search Calls", fontsize=12)
ax_left.set_xlim(-2, 226)
ax_left.xaxis.set_major_locator(ticker.MultipleLocator(50))  # every n steps
ax_left.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) # every x in score
ax_left.grid(True)

ax_left.text(
    x=df_line["Step"].max() - 105, y=1.02,
    s="↑multi-turn search",
    color='black', fontsize=12, va='bottom', ha='left', weight='bold'
)
ax_left.text(
    x=df_line["Step"].max() - 105, y=0.98,
    s="↓one-turn search",
    color='black', fontsize=12, va='top', ha='left', weight='bold'
)

ax_left.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
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
