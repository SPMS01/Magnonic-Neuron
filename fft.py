import os, glob
import numpy as np
import matplotlib.pyplot as plt
import utils
import re

INPUT_DIR = "neuron.out"
DT = 50e-12
Y_RANGE = (0, 20)
X_RANGE = (500, 501)
FILE_PREFIX = "m_full"
# Y_RANGE = (0, 5)
# X_RANGE = (200, 201)

x_t = []

for i in range(len([name for name in os.listdir(INPUT_DIR) if re.fullmatch(rf"{FILE_PREFIX}\d+\.npy", name)])):
    # x magnetisation, layer 0 for z (1 layer), y slice, x range
    data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}{i:06d}.npy'))[utils.Dimension.X.value, 0, Y_RANGE[0]:Y_RANGE[1], X_RANGE[0]:X_RANGE[1]].mean()
    x_t.append(data)

x_t = np.array(x_t)  # (Nt,)
s = x_t - x_t.mean()

S = np.fft.fft(s)
freq = np.fft.fftfreq(len(s), DT)

pos = freq > 0
freq = freq[pos]
power = np.abs(S[pos])**2

# print highest peak
peak_idx = np.argmax(power)
print(f"Highest peak at {freq[peak_idx]/1e9:.2f} GHz with power {power[peak_idx]:.2e}")

plt.plot(freq / 1e9, power)
plt.xlabel("Frequency (GHz)")
plt.ylabel("FFT power")
plt.xlim(6, 8)
plt.tight_layout()
plt.savefig("improved_neuron_fft.png")