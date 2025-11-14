import detector
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, peak_widths, savgol_filter, butter, filtfilt
import os

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
    total_energy = np.sum(Mx_post_coupler**2 + My_post_coupler**2)

    if debug:
        # Mx_input = region_data[utils.DetectorRegionType.INPUT][:, 0]
        # My_input = region_data[utils.DetectorRegionType.INPUT][:, 1]
        # P_input_avg = np.mean(Mx_input**2 + My_input**2)
        # print(f"Average input power: {P_input_avg:.6g} a.u.")

        # Mx_pre_coupler = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 0]
        # My_pre_coupler = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 1]
        # P_pre_coupler_avg = np.mean(Mx_pre_coupler**2 + My_pre_coupler**2)
        # print(f"Average pre-coupler power: {P_pre_coupler_avg:.6g} a.u.")

        Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
        My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
        P_post_coupler_avg = np.mean(Mx_post_coupler**2 + My_post_coupler**2)
        print(f"Average post-coupler power: {P_post_coupler_avg:.6g} a.u.")

        # transmission_ratio = P_post_coupler_avg / P_input_avg
        # print(f"Transmission Ratio (Post-Coupler/Input): {transmission_ratio:.6g}")

        time_axis = np.arange(len(Mx_post_coupler)) * dt
        
        # plot_signal(time_axis, Mx_input**2 + My_input**2, "Input Region Signal Energy Over Time", "Energy (a.u.)", "input_region_signal_energy.png")
        # plot_signal(time_axis, Mx_pre_coupler**2 + My_pre_coupler**2, "Pre-Coupler Output Region Signal Energy Over Time", "Energy (a.u.)", "pre_coupler_output_region_signal_energy.png")
        plot_signal(time_axis, Mx_post_coupler**2 + My_post_coupler**2, "Post-Coupler Output Region Signal Energy Over Time", "Energy (a.u.)", "post_coupler_output_region_signal_energy.png")

    return total_energy