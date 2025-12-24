import os, glob
import numpy as np
import matplotlib.pyplot as plt
import utils
import re

prefix = "paper_coupler" # was paper_coupler
postfix = "sinc" # typically {f}GHz_{m}mT

INPUT_DIR = "paper_coupler_sinc_160ns.out"
DT = 50e-12
Y_SLICE = 3
X_RANGE = (600, 3600)
FILE_PREFIX = "m_full"

x_t = []

for i in range(len([name for name in os.listdir(INPUT_DIR) if re.fullmatch(rf"{FILE_PREFIX}\d+\.npy", name)])):
    # x magnetisation, layer 0 for z (1 layer), singular y slice, x range
    data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}{i:06d}.npy'))[utils.Dimension.X.value, 0, Y_SLICE:Y_SLICE+1, X_RANGE[0]:X_RANGE[1]][0]
    x_t.append(data)

x_t = np.array(x_t) # (Nt, Nx)

# remove dc component
s = x_t.copy()
s = s - s.mean(axis=0, keepdims=True)

# windowing
Nt, Nx = s.shape
wt = np.hanning(Nt)[:, None]
wx = np.hanning(Nx)[None, :]
sw = s * wt * wx

# 2d fft -> f-k
S = np.fft.fftshift(np.fft.fft2(sw))
I = np.abs(S)**2

# axes
dx = 20e-9
f = np.fft.fftshift(np.fft.fftfreq(Nt, d=DT)) / 1e9 # GHz
k = (np.fft.fftshift(np.fft.fftfreq(Nx, d=dx)) * 2*np.pi) / 1e6  # rad/µm

# keep positive frequencies for plotting
mask = f > 0
f_pos = f[mask]
I_pos = I[mask, :]

plt.figure(figsize=(7,4))
plt.pcolormesh(k, f_pos, np.log10(I_pos + 1e-30), shading="auto", cmap="inferno")
plt.xlabel("k (rad/µm)") # Labeled correctly now
plt.ylabel("f (GHz)")
plt.ylim(5.5, 7.5) # (5.5, 7.5)
plt.xlim(-30, 0) # (-30, 0)
plt.title(f"2D spectral map ({INPUT_DIR})")
plt.colorbar(label="log10 intensity")
plt.tight_layout()
plt.savefig(f"peepeepoopoo.png") #paper_f-k_plot_{postfix}.png
plt.show()
plt.close()