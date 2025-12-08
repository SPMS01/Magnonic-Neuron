import detector
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, peak_widths, savgol_filter, butter, filtfilt
import os
import scipy

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

def evaluate_objective(detector_regions: list[utils.DetectorRegion], 
                       input_dir: str, 
                       frame_count: int, 
                       dt: int, 
                       debug: bool = False):
    """
    The objective function just evaluates how well the design does; you can set it as whatever you want

    Args:
        detector_regions (list[utils.DetectorRegion]): List of detector regions to evaluate. Since its a list you can just add as many regions as you want and reference them in this function. Just make sure you actually pass them in.
        input_dir (str): Directory containing the simulation output files (in numpy format).
        frame_count (int): Number of frames in the simulation output.
        dt (int): Time step between frames.
        debug (bool): If True, print debug information and plot signals.
    """
    region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

    for detector_region in detector_regions:
        x_range = detector_region.x_range
        y_range = detector_region.y_range

        wave_data = detector.detect_waves(
            x_range=x_range,
            y_range=y_range,
            input_dir=input_dir,
            frame_count=frame_count,
            dt=dt,
            detector_name=detector_region.region_type.name.lower(),
            debug=debug
        )

        region_data[detector_region.region_type] = wave_data

    Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
    My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
    """
    Since the goal (so far) is just to maximise energy output over time, we just need to sum up the total energy over time.
    A = sqrt(Ax^2 + Ay^2)
    => E ∝ A^2 => E ∝ Ax^2 + Ay^2
    """
    total_energy = np.sum(Mx_post_coupler**2 + My_post_coupler**2)

    if debug:
        Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
        My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
        P_post_coupler_avg = np.mean(Mx_post_coupler**2 + My_post_coupler**2)
        print(f"Average post-coupler power: {P_post_coupler_avg:.6g} a.u.")

        time_axis = np.arange(len(Mx_post_coupler)) * dt

        plot_signal(time_axis, Mx_post_coupler**2 + My_post_coupler**2, "Post-Coupler Output Region Signal Energy Over Time", "Energy (a.u.)", "post_coupler_output_region_signal_energy.png")

    return total_energy

if __name__ == "__main__":
    INPUT_DIR = './ring-resonator.out'

    pre_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
        x_range=(199, 200),
        y_range=(0, 15)
    )

    post_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
        x_range=(719, 720),
        y_range=(0, 15)
    )

    region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

    for detector_region in [pre_coupler_output_detector_region, post_coupler_output_detector_region]:
        x_range = detector_region.x_range
        y_range = detector_region.y_range

        wave_data = detector.detect_waves(
            x_range=x_range,
            y_range=y_range,
            input_dir=INPUT_DIR,
            frame_count=len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')]),
            dt=50e-12,
            detector_name=detector_region.region_type.name.lower(),
            debug=False
        )

        region_data[detector_region.region_type] = wave_data

    Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
    My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
    t = np.arange(len(Mx_post_coupler)) * 50e-12

    cut = 0
    mask = t > cut
    t = t[mask]
    mx = Mx_post_coupler[mask]
    my = My_post_coupler[mask]
    f0 = 2.5e9

    amplitude, phase, complex_bb = extract_vector_frequency_amplitude(mx, my, 50e-12, f0, 10e-9)

    plot_signal(
        time_axis=t,
        signal=amplitude,
        title=f"Post-Coupler Output Region Transverse Magnetisation Amplitude Over Time ({f0*1e-9:.2f} GHz Component)",
        ylabel="|M⊥| (a.u.)",
        filename=f"post_coupler_output_region_transverse_magnetisation_amplitude_{f0*1e-9:.2f}GHz.png"
    )

    plot_signal(
        time_axis=np.arange(len(Mx_post_coupler)) * 50e-12,
        signal=np.sqrt(Mx_post_coupler**2 + My_post_coupler**2),
        title="Post-Coupler Output Region Transverse Magnetisation Amplitude Over Time",
        ylabel="|M⊥| (a.u.)",
        filename="post_coupler_output_region_transverse_magnetisation_amplitude.png"
    )

    Yx = np.fft.rfft(mx - np.mean(mx))
    Yy = np.fft.rfft(my - np.mean(my))
    freqs = np.fft.rfftfreq(len(mx), d=50e-12)

    power = np.abs(Yx)**2 + np.abs(Yy)**2

    print(f"Top 5 frequencies (GHz) by power: {freqs[np.argsort(power)[-5:]][::-1]*1e-9}")

    plt.figure()
    plt.plot(freqs*1e-9, power)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (a.u.)")
    plt.title("FFT of Post-Coupler Output")
    plt.xlim(0, 5)  # focus on low GHz range
    plt.savefig("post_coupler_output_region_fft.png")
    plt.close()