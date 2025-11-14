from utils import Dimension
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import os
import matplotlib.pyplot as plt
from utils import Dimension

def detect_waves(x_range: tuple, 
                 y_range: tuple, 
                 input_dir: str, 
                 frame_count: int, 
                 dt: int, 
                 detector_name: str = "detector",
                 debug: bool = False) -> np.ndarray:
    x_start, x_end = x_range
    y_start, y_end = y_range
    wave_data_x = []
    wave_data_y = []

    for i in range(frame_count):
        data = np.load(os.path.join(input_dir, f'm{i:06d}.npy'))

        # Each .npy file stores [Mx, My, Mz] components
        slice_x = data[0, 0, y_start:y_end, x_start:x_end]
        slice_y = data[1, 0, y_start:y_end, x_start:x_end]

        avg_x = np.mean(slice_x)
        avg_y = np.mean(slice_y)

        wave_data_x.append(avg_x)
        wave_data_y.append(avg_y)

    # Convert to NumPy arrays
    wave_data_x = np.array(wave_data_x)
    wave_data_y = np.array(wave_data_y)

    # Combine both into shape (frame_count, 2)
    wave_data = np.stack((wave_data_x, wave_data_y), axis=1)

    if debug:
        # Plot both Mx and My signals
        # plt.figure(figsize=(10, 5))
        # plt.plot(wave_data_x, label='Mx')
        # plt.plot(wave_data_y, label='My')
        # plt.title("Average Magnetisation (X and Y axes) at Detector")
        # plt.xlabel("Frame")
        # plt.ylabel("Magnetisation")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(input_dir, "../detector_signal_xy.png"))

        # Also plot the amplitude (optional)
        amplitude = np.sqrt(wave_data_x**2 + wave_data_y**2)
        plt.figure(figsize=(10, 5))
        plt.plot(amplitude, color='purple')
        plt.title("Transverse Magnetisation Amplitude |M⊥|")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(input_dir, f"../{detector_name}_amplitude.png"))

    return wave_data

if __name__ == "__main__":
    # === Sanity check ===
    INPUT_DIR = './ring-resonator.out'
    end_detector_start = (719, 0)
    end_detector_end = (720, 15)

    start_detector_start = (199, 0)
    start_detector_end = (200, 15)

    # Design region is 1um * 1um so i dont set the end
    design_region_x_start = 50
    design_region_y_start = 16

    input_detector_start = (199, 67)
    input_detector_end = (200, 82)

    data = np.load(os.path.join(INPUT_DIR, f'm000400.npy'))[Dimension.X.value, 0]

    # === PLOT ===
    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

    # Draw detector boxes
    plt.axvline(end_detector_start[0], color='lime', linestyle='--')
    plt.axvline(end_detector_end[0], color='lime', linestyle='--')
    plt.axhline(end_detector_start[1], color='lime', linestyle='--')
    plt.axhline(end_detector_end[1], color='lime', linestyle='--')
    plt.text(end_detector_start[0], end_detector_end[1]+1, 'Post-coupler Detector Region', color='lime')

    plt.axvline(start_detector_start[0], color='cyan', linestyle='--')
    plt.axvline(start_detector_end[0], color='cyan', linestyle='--')
    plt.axhline(start_detector_start[1], color='cyan', linestyle='--')
    plt.axhline(start_detector_end[1], color='cyan', linestyle='--')
    plt.text(start_detector_start[0], start_detector_end[1]+1, 'Pre-coupler Detector Region', color='cyan')

    plt.axvline(input_detector_start[0], color='orange', linestyle='--')
    plt.axvline(input_detector_end[0], color='orange', linestyle='--')
    plt.axhline(input_detector_start[1], color='orange', linestyle='--')
    plt.axhline(input_detector_end[1], color='orange', linestyle='--')
    plt.text(input_detector_start[0], input_detector_end[1]+1, 'Input Detector Region', color='orange')

    # Detector box for design region
    # plt.axvline(design_region_x_start, color='red', linestyle='--')
    # plt.axvline(design_region_x_start + 50, color='red', linestyle='--')
    # plt.axhline(design_region_y_start, color='red', linestyle='--')
    # plt.axhline(design_region_y_start + 50, color='red', linestyle='--')
    # plt.text(design_region_x_start, design_region_y_start+1, 'Design Region Detector Region', color='red')

    plt.xlabel('X index')
    plt.ylabel('Y index')
    plt.tight_layout()
    # plt.show()
    plt.savefig("sanity_check_plot.png")