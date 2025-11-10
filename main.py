from detector import detect_waves
import objective_function
import utils
import numpy as np
import os

M0 = np.random.choice([0, 1], size=(50, 50))
INPUT_DIR = './ring-resonator.out'
frame_count = len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')])

print(frame_count)
# detect_waves(
#     x_range=(130, 131),
#     y_range=(0, 15),
#     input_dir='./ring-resonator.out',
#     frame_count=8001,
#     dt=50e-12,
#     dimension=Dimension.X,
#     debug=True
# )

input_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.INPUT,
    x_range=(199, 200),
    y_range=(83, 98)
)

pre_coupler_output_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
    x_range=(199, 200),
    y_range=(16, 31)
)

post_coupler_output_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    x_range=(719, 720),
    y_range=(16, 31)
)

objective_function.evaluate_objective(
    detector_regions=[pre_coupler_output_detector_region, post_coupler_output_detector_region, input_detector_region],
    input_dir=INPUT_DIR,
    frame_count=frame_count,
    dt=50e-12,
    debug=True
)