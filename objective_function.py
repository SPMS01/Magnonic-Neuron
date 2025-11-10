import detector
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, peak_widths, savgol_filter, butter, filtfilt

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
            debug=debug
        )

        region_data[detector_region.region_type] = wave_data

    Mx_input = region_data[utils.DetectorRegionType.INPUT][:, 0]
    My_input = region_data[utils.DetectorRegionType.INPUT][:, 1]
    P_input_avg = np.mean(Mx_input**2 + My_input**2)
    print(f"Average input power: {P_input_avg:.6g} a.u.")

    Mx_pre_coupler = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 0]
    My_pre_coupler = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 1]
    P_pre_coupler_avg = np.mean(Mx_pre_coupler**2 + My_pre_coupler**2)
    print(f"Average pre-coupler power: {P_pre_coupler_avg:.6g} a.u.")

    Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
    My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
    P_post_coupler_avg = np.mean(Mx_post_coupler**2 + My_post_coupler**2)
    print(f"Average post-coupler power: {P_post_coupler_avg:.6g} a.u.")

    transmission_ratio = P_post_coupler_avg / P_input_avg
    print(f"Transmission Ratio (Post-Coupler/Input): {transmission_ratio:.6g}")
    
    # P_pre_out = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT] ** 2
    # print(f"Average pre-coupler output power: {np.mean(P_pre_out):.6g} a.u.")

    # P_post_out = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT] ** 2
    # print(f"Average post-coupler output power: {np.mean(P_post_out):.6g} a.u.")

    # transmission_ratio = np.mean(P_post_out) / np.mean(P_pre_out)
    # print(f"Transmission Ratio (Post/Pre): {transmission_ratio:.6g}")

    # time_axis = np.arange(len(Mx_post_coupler)) * dt
    
    # if debug:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(time_axis * 1e9, P_post_coupler_avg, label='Output Region Signal Energy', color='orange')
    #     plt.title('Output Region Signal Energy Over Time')
    #     plt.xlabel('Time (ns)')
    #     plt.ylabel('Energy (a.u.)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig("output_region_signal_energy.png")