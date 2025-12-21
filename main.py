from detector import detect_waves
import objective_function
import utils
import numpy as np
import os
import dbs
import shutil

# Initial random design of YIG and air
# M0 = np.random.choice([0, 1], size=(50, 50))

N = 50
y, x = np.indices((N, N))
cx = cy = (N - 1) / 2
outer_r, inner_r = 25, 20
dist = np.sqrt((x - cx)**2 + (y - cy)**2)
M0 = ((dist <= outer_r) & (dist >= inner_r)).astype(np.uint8)

MX3_EXE_PATH = 'mumax3'
MX3_EXE_CONVERT_PATH = 'mumax3-convert'
INPUT_DIR = './ring-resonator.out'
OUTPUT_DIR = './dbs_output'
TEMPLATE_PATH = 'resonator_template.mx3'

# Clean output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

M_output, final_score = dbs.direct_binary_search_decay(
    M0=M0,
    mx3_exe_path=MX3_EXE_PATH,
    mx3_exe_convert_path=MX3_EXE_CONVERT_PATH,
    template_path=TEMPLATE_PATH,
    output_dir=OUTPUT_DIR,
    initial_patch_size=3,
    max_iterations=50,
    tolerance=0.01
)

print(f"Final score: {final_score:.6g}")

# frame_count = len([name for name in os.listdir(INPUT_DIR) if name.endswith('.npy')])

# print(frame_count)

# input_detector_region = utils.DetectorRegion(
#     region_type=utils.DetectorRegionType.INPUT,
#     x_range=(199, 200),
#     y_range=(67, 82)
# )

# pre_coupler_output_detector_region = utils.DetectorRegion(
#     region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
#     x_range=(199, 200),
#     y_range=(0, 15)
# )

# post_coupler_output_detector_region = utils.DetectorRegion(
#     region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
#     x_range=(719, 720),
#     y_range=(0, 15)
# )

# objective_function.evaluate_objective(
#     detector_regions=[pre_coupler_output_detector_region, post_coupler_output_detector_region, input_detector_region],
#     input_dir=INPUT_DIR,
#     frame_count=frame_count,
#     dt=50e-12,
#     debug=True
# )