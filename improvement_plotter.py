import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

INPUT_CSV = 'compiled dbs changes.csv'

df = pd.read_csv(INPUT_CSV)
df['Flipped'] = df['Flipped'].astype(str).str.upper() == 'TRUE'

flipped_df = df[df['Flipped']]

iteration_regions = [
    (0, 100, r"$\mathrm{Iter.\ 1\ (100\ nm)}$"),
    (100, 200, r"$\mathrm{Iter.\ 2\ (100\ nm)}$"),
    (200, 825, r"$\mathrm{Iter.\ 1\ (40\ nm)}$"),
    (825, 1450, r"$\mathrm{Iter.\ 2\ (40\ nm)}$"),
    (1450, 2075, r"$\mathrm{Iter.\ 3\ (40\ nm)}$")
] 

matplotlib.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

plt.figure()
ax = plt.gca()

# set plot size
# ax.figure.set_size_inches(5, 5)

ax.scatter(flipped_df['Flip'], flipped_df['Score']/1e11, marker='o', color='black')
ax.plot(flipped_df['Flip'], flipped_df['Score']/1e11, linestyle='-', color='black')
ax.set_xlabel(r"$\mathrm{Flip \ Number}$", fontsize=20)
ax.set_ylabel(r"$\mathrm{Reward \ Function} \ R \ \mathrm{(10^{11})}$", fontsize=20)
ax.tick_params(labelsize=16, width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.ylim(top=2.2)

ax.text(
    x=200 - 50,
    y=ax.get_ylim()[1] - 0.05,
    s=r"$\mathrm{100 \ nm}$",
    rotation=90,
    va="top",
    ha="right",
    fontsize=16
)

ax.text(
    x=200 + 50,
    y=ax.get_ylim()[1] - 0.05,
    s=r"$\mathrm{40 \ nm}$",
    rotation=90,
    va="top",
    ha="left",
    fontsize=16
)

shaded_areas = [(100, 200), (825, 1450), (2075, 2500)]
 
for start, end in shaded_areas:
    ax.fill_betweenx(
    y=[ax.get_ylim()[0], ax.get_ylim()[1]],
    x1=start,
    x2=end,
    color='lightgray',
    alpha=0.5,
    zorder=0
)

ax.axvline(x=200, color='black', linestyle='--', linewidth=1.5)

iteration_ticks = [50, 150, 512.5, 1137.5, 1762.5, 2325]
iteration_labels = ['1', '2', '1', '2', '3', '4']

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())

ax_top.set_xticks(iteration_ticks)
ax_top.set_xticklabels(iteration_labels, fontsize=18)

ax_top.set_xlabel(r"$\mathrm{Iteration}$", fontsize=20, labelpad=10)
ax_top.tick_params(axis='x', length=0)

for spine in ax_top.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('improvement_plot.svg')
plt.close()