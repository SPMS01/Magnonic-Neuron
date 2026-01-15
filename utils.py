import subprocess
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

class DetectorRegionType(Enum):
    DESIGN = 0
    POST_COUPLER_OUTPUT = 1
    PRE_COUPLER_OUTPUT = 2
    INPUT = 3
    RING_WAVEGUIDE_INTERSECTION_OUTPUT = 4
    EXCITATION_REGION = 5,
    OTHER = 99

class DetectorRegion:
    def __init__(self, region_type: DetectorRegionType, x_range: tuple, y_range: tuple) -> None:
        self.region_type = region_type
        self.x_range = x_range
        self.y_range = y_range

def generate_mx3_design(M: np.ndarray, output_path: str, template_path: str, x_offset=0, y_offset=0, height=50) -> None:
    """
    Generate a Mumax3 script from the given magnetisation matrix M.
    
    Args:
        M (np.ndarray): Magnetisation matrix.
        output_path (str): Path to save the generated Mumax3 script.
        template_path (str): Path to the Mumax3 template file.
        X_OFFSET (int): X offset for the design region.
        HEIGHT (int): Height of the design region.
    """
    # this is just like numpy; (0, 0) is bottom-left
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), template_path), "r") as template_file:
        content = template_file.read()

    magnetisation = ""
    for i in range(M.shape[0]):     # i = y (down in NumPy)
        for j in range(M.shape[1]): # j = x (right in NumPy)
            region = M[i, j]
            x = x_offset + j
            y = y_offset + height - 1 - i  # flip vertically for Mumax3
            magnetisation += f"DefRegionCell({region}, {x}, {y}, 0)\n"
    
    content = content.replace("// {{ INSERT CELL CODE HERE }}", magnetisation)

    with open(output_path, "w+") as output_file:
        output_file.seek(0)
        output_file.write(content)

def run_mx3(mx3_exe_path: str, mx3_exe_convert_path: str, mx3_file_path: str, output_dir: str):
    """
    Run the Mumax3 simulation and convert the output to NumPy format.
    
    Args:
        mx3_exe_path (str): Path to the Mumax3 executable.
        mx3_exe_convert_path (str): Path to the Mumax3 convert executable.
        mx3_file_path (str): Path to the Mumax3 script file.
        output_dir (str): Directory where the output files will be saved.
    
    Returns:
        dict: Contains stdout, stderr, and return code of the Mumax3 execution and conversion.
    """
    mx3_result = subprocess.run([mx3_exe_path, mx3_file_path], capture_output=True, text=True)
    
    if mx3_result.returncode == 0:
        mx3_convert_result = subprocess.run(
            f"{mx3_exe_convert_path} -numpy {output_dir}/**/*.ovf",
            shell=True, capture_output=True, text=True
        )
    else:
        mx3_convert_result = None
        print(mx3_result.stderr)
        exit(0)

    return {
        'mumax3_stdout': mx3_result.stdout,
        'mumax3_stderr': mx3_result.stderr,
        'mumax3_returncode': mx3_result.returncode,
        'convert_stdout': mx3_convert_result.stdout if mx3_convert_result else '',
        'convert_stderr': mx3_convert_result.stderr if mx3_convert_result else '',
        'convert_returncode': mx3_convert_result.returncode if mx3_convert_result else -1
    }

def plot_signal(time_axis: np.ndarray, signal: np.ndarray, title: str, ylabel: str, filename: str):
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis * 1e9, signal, label=ylabel)
    plt.title(title)
    plt.xlabel('Time (ns)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def extract_vector_frequency_amplitude(mx: np.ndarray, my: np.ndarray, dt: float, f0: float, window: float=10e-9):
    """
    Extract instantaneous amplitude and relative phase of a given frequency
    component from two transverse components mx(t), my(t).

    Parameters
    ----------
    mx, my : np.ndarray
        Real-valued time series arrays for m_x(t) and m_y(t).
    dt : float
        Time step between samples in seconds.
    f0 : float
        Target frequency in Hz (e.g. 1.5e9).
    window : float
        Width (s) of moving-average low-pass filter used on the baseband.
        Larger -> smoother envelope.
    """

    assert len(mx) == len(my), "Input signals must have the same length."
    t = np.arange(len(mx)) * dt

    # Form the complex vector signal and remove its DC component
    m_complex = (mx - np.mean(mx)) + 1j * (my - np.mean(my))

    # demodulate to baseband
    exp = np.exp(-1j * 2 * np.pi * f0 * t)
    m_bb = m_complex * exp

    # moving-average low-pass filter
    sigma_samples = (window / dt) / 6
    if sigma_samples < 1:
        sigma_samples = 1

    m_bb_filtered = scipy.ndimage.gaussian_filter1d(m_bb, sigma=sigma_samples, mode='nearest')
    m_bb_filtered *= 2

    amplitude = np.abs(m_bb_filtered)
    phase = np.angle(m_bb_filtered)

    return amplitude, phase, m_bb_filtered

def plot_detector_regions(reference_frame: str, regions: list[DetectorRegion], output_path: str):
    data = np.load(reference_frame)[Dimension.X.value, 0]  # x-component
    
    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][0]}]')

    for region in regions:
        rect = plt.Rectangle(
            (region.x_range[0], region.y_range[0]),
            region.x_range[1] - region.x_range[0],
            region.y_range[1] - region.y_range[0],
            linewidth=0.5, edgecolor='lime', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            region.x_range[0],
            region.y_range[1] + 1,
            region.region_type.name,
            color='lime',
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='left',
            weight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # # detector region
    # rect = plt.Rectangle(
    #     (region.x_range[0], region.y_range[0]),
    #     region.x_range[1] - region.x_range[0],
    #     region.y_range[1] - region.y_range[0],
    #     linewidth=2, edgecolor='lime', facecolor='none'
    # )
    # plt.gca().add_patch(rect)

    # plt.tight_layout()
    # plt.savefig(output_path)
    # plt.close()