import utils
import detector
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_DIR = "./coupler.out"

pre_coupler_input_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
    x_range=(149, 150),
    y_range=(0, 15)
)

post_coupler_output_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    x_range=(550, 551),
    y_range=(0, 15)
)

region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

for detector_region in [pre_coupler_input_region, post_coupler_output_region]:
    x_range = detector_region.x_range
    y_range = detector_region.y_range

    wave_data = detector.detect_waves(
        x_range=x_range,
        y_range=y_range,
        input_dir=INPUT_DIR,
        frame_count=len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')]),
        dt=50e-12,
        debug=False
    )

    region_data[detector_region.region_type] = wave_data

mx = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 0]
my = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 1]

Yx = np.fft.rfft(mx - np.mean(mx))
Yy = np.fft.rfft(my - np.mean(my))
freqs = np.fft.rfftfreq(len(mx), 50e-12)
power = np.abs(Yx)**2 + np.abs(Yy)**2

plt.figure()
plt.plot(freqs * 1e-9, power)
plt.xlim(0, 5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (a.u.)")
plt.title("Post-Coupler Output Region Frequency Spectrum")
plt.savefig("post_coupler_output_region_frequency_spectrum.png")
plt.close()