import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from detector import detect_waves
import os
import re

INPUT_DIR = "10mTmx3runs"
INPUT_RUNS = [
    "initial_design.out",
    "100nm.out",
    "40nmlatest.out"
]
INPUT_TITLES = [
    "Initial Design",
    "100x100 nm",
    "40x40 nm"
]
# black, dark red, dark blue
COLORS = ["black", "darkred", "darkblue"]
FILE_PREFIX = "m_full"
DT = 50e-12

matplotlib.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

plt.figure()
plt.ylim(0, 33000)
ax = plt.gca()

for run, title, color in zip(INPUT_RUNS, INPUT_TITLES, COLORS):
    data = detect_waves(
        x_range=(200, 201),
        y_range=(0, 20),
        input_dir=f"{INPUT_DIR}/{run}",
        frame_count=len([name for name in os.listdir(f"{INPUT_DIR}/{run}") if re.fullmatch(rf"{FILE_PREFIX}\d+\.npy", name)]),
        dt=DT,
        file_prefix=FILE_PREFIX,
        detector_name=f"{INPUT_DIR}/{run}detector_1",
        debug=False
    )

    amplitudes = np.sqrt(data[:,0]**2 + data[:,1]**2)
    time = np.arange(len(amplitudes)) * DT * 1e9  # Convert to ns

    ax.plot(
        time,
        amplitudes,
        linestyle='-',
        label=title,
        color=color
    )

ax.axhline(y=9350, color="black", linestyle='--', linewidth=1.5)
ax.text(
    x=ax.get_xlim()[0],
    y=9350,
    s=r"$\langle M_\perp \rangle \mathrm{\ for \ 10 \ mT}$",
    va='bottom',
    ha='left',
    fontsize=16
)

ax.axhline(y=29037, color="black", linestyle='--', linewidth=1.5)
ax.text(
    x=ax.get_xlim()[0],
    y=29037,
    s=r"$\langle M_\perp \rangle \mathrm{\ for \ 33 \ mT}$",
    va='bottom',
    ha='left',
    fontsize=16
)

ax.set_xlabel(r"$\mathrm{Time \ (ns)}$", fontsize=20)
ax.set_ylabel(r"$\langle M_\perp \rangle \ \mathrm{\ (A/m)}$", fontsize=20)
ax.tick_params(labelsize=16, width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig('amplitude_improvement_plot.svg')
plt.close()