import utils
import pathlib
import detector
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_DIR = './000008_design.out'
OUTPUT_DIR = 'neuron_functionality_test_output'
DT = 50e-12

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

post_coupler_output_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    x_range=(719, 720),
    y_range=(0, 15)
)

ring_waveguide_intersection_output_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.RING_WAVEGUIDE_INTERSECTION_OUTPUT,
    x_range=(150, 200),
    y_range=(0, 15)
)

excitation_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.EXCITATION_REGION,
    x_range=(305, 310),
    y_range=(67, 82)
)

regions = [post_coupler_output_detector_region, 
           ring_waveguide_intersection_output_detector_region,
           excitation_region]
region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

for detector_region in regions:
    x_range = detector_region.x_range
    y_range = detector_region.y_range

    wave_data = detector.detect_waves(
        x_range=x_range,
        y_range=y_range,
        input_dir=INPUT_DIR,
        frame_count=len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')]),
        dt=DT,
        detector_name=detector_region.region_type.name.lower(),
        debug=False
    )

    region_data[detector_region.region_type] = wave_data

# Plot detector regions
utils.plot_detector_regions(
    os.path.join(INPUT_DIR, "m001600.npy"),
    regions,
    os.path.join(OUTPUT_DIR, "regions.png")
)

# Plot the signal energy over 
for region in regions:
    Mx = region_data[region.region_type][:, 0]
    My = region_data[region.region_type][:, 1]
    time_axis = np.arange(len(Mx)) * DT

    utils.plot_signal(
        time_axis,
        Mx**2 + My**2,
        f"{region.region_type.name.replace('_', ' ').title()} Signal Energy Over Time",
        "Energy (a.u.)",
        os.path.join(OUTPUT_DIR, f"{region.region_type.name.lower()}_signal_energy.png")
    )
    
# Plot FFT spectrum
Mx_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
My_post_coupler = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]

Yx = np.fft.rfft(Mx_post_coupler - np.mean(Mx_post_coupler))
Yy = np.fft.rfft(My_post_coupler - np.mean(My_post_coupler))
freqs = np.fft.rfftfreq(len(Mx_post_coupler), DT)
power = np.abs(Yx)**2 + np.abs(Yy)**2
top_freqs = freqs[np.argsort(power)[-5:]][::-1]
print(f"Top frequency components (GHz): {top_freqs}")

plt.figure()
plt.plot(freqs * 1e-9, power)
plt.xlim(0, 5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (a.u.)")
plt.title("Post-Coupler Output Region Frequency Spectrum")
plt.savefig(os.path.join(OUTPUT_DIR, "post_coupler_output_region_frequency_spectrum.png"))
plt.close()

# Get energy graph of top frequency
amplitude, phase, m_bb = utils.extract_vector_frequency_amplitude(
    Mx_post_coupler - np.mean(Mx_post_coupler),
    My_post_coupler - np.mean(My_post_coupler),
    dt=DT,
    f0=top_freqs[0]
)
utils.plot_signal(
    time_axis,
    amplitude,
    f"Post-Coupler Output Region Signal Energy at {top_freqs[0]*1e-9:.2f}GHz Over Time",
    "Energy (a.u.)",
    os.path.join(OUTPUT_DIR, f"post_coupler_output_region_signal_energy_{top_freqs[0]*1e-9:.2f}GHz.png")
)