import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from utils import Dimension

FILE_PREFIX = "m"
INPUT_DIR = './ring-resonator.out'
FRAME_COUNT = 2001
DIMENSION = Dimension.X

# === Precompute color scale ===
print("Precomputing vabs...")
vabs = 0
for i in range(FRAME_COUNT):
    path = os.path.join(INPUT_DIR, f"{FILE_PREFIX}{i:06d}.npy")
    if os.path.exists(path):
        data = np.load(path)[DIMENSION.value, 0]
        vabs = max(vabs, np.max(np.abs(data)))
print(f"Using ±{vabs:.3g} as color range")

# === Setup plot ===
fig, ax = plt.subplots(figsize=(10, 4))
img = ax.imshow(np.zeros((1, 1)), cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower', aspect='auto')
cbar = fig.colorbar(img, ax=ax, label='Magnetisation (arb. units)')
title = ax.set_title("Frame 0")
ax.set_xlabel("X index")
ax.set_ylabel("Y index")

# === Update function ===
def update(frame):
    path = os.path.join(INPUT_DIR, f"{FILE_PREFIX}{frame:06d}.npy")
    if not os.path.exists(path):
        print(f"Missing frame {frame}")
        return [img]
    data = np.load(path)[DIMENSION.value, 0]
    img.set_data(data)
    title.set_text(f"Frame {frame} — m[{['x', 'y', 'z'][DIMENSION.value]}]")
    return [img]

# === Animate & Save ===
print("Rendering animation...")
ani = animation.FuncAnimation(fig, update, frames=FRAME_COUNT, blit=True)
ani.save("animation.mp4", fps=30, dpi=150)
print("Saved as animation.mp4")