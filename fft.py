import os, glob
import numpy as np
import matplotlib.pyplot as plt
import utils

INPUT_DIR = "resonator_7.05GHz.out"
DT = 50e-12
Y_RANGE = (59, 65)
X_RANGE = (170, 171)
# Y_RANGE = (0, 5)
# X_RANGE = (200, 201)

x_t = []

for i in range(len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')])):
    # x magnetisation, layer 0 for z (1 layer), y slice, x range
    data = np.load(os.path.join(INPUT_DIR, f'm_full{i:06d}.npy'))[utils.Dimension.X.value, 0, Y_RANGE[0]:Y_RANGE[1], X_RANGE[0]:X_RANGE[1]].mean()
    x_t.append(data)

x_t = np.array(x_t)  # (Nt,)
s = x_t - x_t.mean()

S = np.fft.fft(s)
freq = np.fft.fftfreq(len(s), DT)

pos = freq > 0
freq = freq[pos]
power = np.abs(S[pos])**2

plt.plot(freq / 1e9, power)
plt.xlabel("Frequency (GHz)")
plt.ylabel("FFT power")
plt.tight_layout()
plt.savefig("resonator_7.05GHz_input.png")