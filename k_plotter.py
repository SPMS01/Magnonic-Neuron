import numpy as np
import os
import matplotlib.pyplot as plt

def spatial_fft_1d(y_line, dx, label):
    N = len(y_line)
    windowed = y_line * np.hanning(N)       # optional Hanning window
    Y = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, d=dx)       # cycles/m
    k_vals = 2 * np.pi * freqs             # rad/m
    mag = np.abs(Y)

    # Skip DC for peak detection
    idx_peak = np.argmax(mag[1:]) + 1
    idx_peak2 = np.argsort(mag[1:])[-2] + 1

    k_peak = k_vals[idx_peak]
    k_peak2 = k_vals[idx_peak2]

    print(f"{label}: dominant k = {k_peak:.2e} rad/m")
    print(f"{label}: second dominant k = {k_peak2:.2e} rad/m")
    print(f"{label}: delta k = {abs(k_peak - k_peak2):.2e} rad/m")

    # Plot
    plt.figure(figsize=(6,3))
    plt.plot(k_vals, mag, label='FFT magnitude')
    plt.plot(k_vals[idx_peak], mag[idx_peak], 'ro', label='Dominant k')
    plt.plot(k_vals[idx_peak2], mag[idx_peak2], 'go', label='Second Dominant k')
    plt.title(f"Spatial FFT ({label})")
    plt.xlabel("k (rad/m)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"spatial_fft_{label}.png")
    plt.close()  # closes figure to avoid overlapping plots in loops

    return k_peak


INPUT_DIR = "./coupler.out"
DX = 20e-9
waveguide_start = ()
waveguide_end = ()
coupler_start = ()
coupler_end = ()

data = np.load(os.path.join(INPUT_DIR, f'm000800.npy'))[0, 0] # x-component

waveguide_line = np.mean(data[23:25, :], axis=0)

k_waveguide = spatial_fft_1d(waveguide_line, DX, "Waveguide")

# === PLOT ===
plt.figure(figsize=(10, 4))
vabs = np.max(np.abs(data))
plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
plt.colorbar(label='Magnetisation (arb. units)')
plt.title(f'm[{["x", "y", "z"][0]}]')

# waveguide detection zone
plt.hlines(23.5, xmin=180, xmax=520, color='lime', linestyle='--') 
plt.text(180, 30, 'Waveguide Detection Zone', color='lime')

# coupler detection zone
# plt.hlines(7.5, xmin=180, xmax=520, color='red', linestyle='--')
# plt.text(180, -10, 'Coupler Detection Zone', color='red')

plt.xlabel('X index')
plt.ylabel('Y index')
plt.tight_layout()
# plt.show()
plt.savefig("k_sanity_plot.png")