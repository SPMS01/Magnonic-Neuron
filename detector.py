from utils import Dimension
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_waves(x_range: tuple, y_range: tuple, input_dir: str, frame_count: int, dt: int, dimension: Dimension):
    x_start, x_end = x_range
    y_start, y_end = y_range
    wave_data = []

    for i in range(frame_count):
        # Load data for the current frame
        data = np.load(os.path.join(input_dir, f'm{i:06d}.npy'))
        slice_2d = data[dimension.value, 0, y_start:y_end, x_start:x_end] # 0 because there's only one Z layer
        avg_value = np.mean(slice_2d)
        wave_data.append(avg_value)

    wave_data = np.array(wave_data)

    # Plot the time-domain signal
    plt.figure(figsize=(10, 5))
    plt.plot(wave_data)
    plt.title(f"Average Magnetisation ({dimension.name} axis) at a Detector")
    plt.xlabel("Frame")
    plt.ylabel("Magnetisation")
    plt.grid(True)
    
    plt.savefig(os.path.join(input_dir, f"../a_fft_signal.png"))

if __name__ == "__main__":
    # detect_waves(
    #     x_range=(224, 225),
    #     y_range=(35, 50),
    #     input_dir='./ring-resonator.out',
    #     frame_count=1001,
    #     dt=50e-12,
    #     dimension=Dimension.X
    # )

    # === Sanity check ===
    INPUT_DIR = './ring-resonator.out'
    x_start, x_end = 130, 131
    y_start, y_end = 0, 15

    data = np.load(os.path.join(INPUT_DIR, f'm001000.npy'))[Dimension.X.value, 0]

    # === PLOT ===
    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

    # Draw detector box
    plt.axvline(x_start, color='lime', linestyle='--')
    plt.axvline(x_end, color='lime', linestyle='--')
    plt.axhline(y_start, color='lime', linestyle='--')
    plt.axhline(y_end, color='lime', linestyle='--')
    plt.text(x_start, y_end+1, 'Detector Region', color='lime')

    plt.xlabel('X index')
    plt.ylabel('Y index')
    plt.tight_layout()
    # plt.show()
    plt.savefig("sanity_check_plot.png")