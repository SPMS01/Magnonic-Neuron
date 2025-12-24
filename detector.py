from utils import Dimension
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import Dimension
import re

def detect_waves(x_range: tuple, 
                 y_range: tuple, 
                 input_dir: str, 
                 frame_count: int, 
                 dt: int,
                 file_prefix: str = "m_full",
                 detector_name: str = "detector",
                 debug: bool = False) -> np.ndarray:
    """
    Grab spin wave data from a specified detector region over time.
    
    Args:
        x_range (tuple): The range of x-coordinates (start, end) for the detector region.
        y_range (tuple): The range of y-coordinates (start, end) for the detector region.
        input_dir (str): Directory containing the simulation output files (in numpy format).
        frame_count (int): Number of frames in the simulation output.
        dt (int): Time step between frames.
        detector_name (str): Name of the detector (used for saving debug plots).
        debug (bool): If True, generate debug plots.

    Returns:
        np.ndarray: Array of shape (frame_count, 2) containing average Mx and My values over time.
    """
    x_start, x_end = x_range
    y_start, y_end = y_range
    wave_data_x = []
    wave_data_y = []
    wave_data_z = []

    for i in range(frame_count):
        data = np.load(os.path.join(input_dir, f'{file_prefix}{i:06d}.npy'))

        # Each .npy file stores [Mx, My, Mz] components
        slice_x = data[0, 0, y_start:y_end, x_start:x_end]
        slice_y = data[1, 0, y_start:y_end, x_start:x_end]
        slice_z = data[2, 0, y_start:y_end, x_start:x_end]

        avg_x = np.mean(slice_x)
        avg_y = np.mean(slice_y)
        avg_z = np.mean(slice_z)

        wave_data_x.append(avg_x)
        wave_data_y.append(avg_y)
        wave_data_z.append(avg_z)

    # Convert to NumPy arrays
    wave_data_x = np.array(wave_data_x)
    wave_data_y = np.array(wave_data_y)
    wave_data_z = np.array(wave_data_z)

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
        # make the x axis in ns
        amplitude = np.sqrt(wave_data_x**2 + wave_data_y**2)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(amplitude)) * dt / 1e-9, amplitude, color='purple')
        plt.title("Transverse Magnetisation Amplitude |M⊥|")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(input_dir, f"../{detector_name}_amplitude.png"))
        plt.close()

        # plot z component
        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(len(wave_data_z)) * dt / 1e-9, wave_data_z, color='green')
        # plt.title("Longitudinal Magnetisation Mz")
        # plt.xlabel("Time (ns)")
        # plt.ylabel("Mz")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(input_dir, f"../{detector_name}_mz.png"))
        # plt.close()

        # # print average of final 20% of mz
        # final_mz_avg = np.mean(wave_data_z[int(0.8 * frame_count):])
        # print(f"Average Mz in final 20% of simulation at detector '{detector_name}': {final_mz_avg}")
        # print(f"Angle: {np.arccos(final_mz_avg) * 180 / np.pi} degrees")
        # print(f"a_0^2: {1 - final_mz_avg}")

    return wave_data

if __name__ == "__main__":
    INPUT_DIR = "neuron.out"
    FILE_PREFIX = "m_full"

    detect_waves(x_range=(500, 501), 
        y_range=(0, 20), 
        input_dir=INPUT_DIR,
        frame_count=len([name for name in os.listdir(INPUT_DIR) if re.fullmatch(rf"{FILE_PREFIX}\d+\.npy", name)]),
        dt=50e-12,
        file_prefix=FILE_PREFIX,
        detector_name="detector_1",
        debug=True)
    
    detect_waves(x_range=(400, 401), 
        y_range=(28, 48), 
        input_dir=INPUT_DIR,
        frame_count=len([name for name in os.listdir(INPUT_DIR) if re.fullmatch(rf"{FILE_PREFIX}\d+\.npy", name)]),
        dt=50e-12,
        file_prefix=FILE_PREFIX,
        detector_name="detector_2",
        debug=True)


    # detect_waves(x_range=(200, 201), 
    #     y_range=(0, 5), 
    #     input_dir=INPUT_DIR,
    #     frame_count=len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')]),
    #     dt=50e-12,
    #     detector_name="umm what the sigma",
    #     debug=True)

    # detect_waves(x_range=(600, 601), 
    #     y_range=(0, 5), 
    #     input_dir=INPUT_DIR,
    #     frame_count=len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')]),
    #     dt=50e-12,
    #     detector_name="umm what the sigma balls",
    #     debug=True)