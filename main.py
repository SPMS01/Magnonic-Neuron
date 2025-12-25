from detector import detect_waves
import objective_function
import utils
import numpy as np
import os
import dbs
import shutil

# Initial random design of YIG and air
# M0 = np.random.choice([0, 1], size=(50, 50))

Nx, Ny = 50, 200
y, x = np.indices((Ny, Nx))
cx = (Nx - 1) / 2
cy = (Ny - 1) / 2
scale_y = Ny / Nx   # = 4
outer_r, inner_r = 25, 17.5  # radius measured in x-pixels (physical units)
dist = np.sqrt(
    (x - cx)**2 +
    ((y - cy) / scale_y)**2
)
M0 = ((dist <= outer_r) & (dist >= inner_r)).astype(np.uint8)

MX3_EXE_PATH = 'mumax3'
MX3_EXE_CONVERT_PATH = 'mumax3-convert'
OUTPUT_DIR = './dbs_output'
TEMPLATE_PATH = 'reduced_neuron_dbs_template.mx3'
DX = 20e-9
DY = 5e-9
# PATCH_SIZE = 100e-9

# Clean output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

m0 = M0.copy()
run_initial = True

for patch_size in [200e-9, 100e-9, 40e-9, 20e-9]:
    output_dir = os.path.join(OUTPUT_DIR, f'patch_{int(patch_size*1e9)}nm')
    os.mkdir(output_dir)

    m, score = dbs.DBS(
        M0=m0,
        mx3_exe_path=MX3_EXE_PATH,
        mx3_exe_convert_path=MX3_EXE_CONVERT_PATH,
        template_path=TEMPLATE_PATH,
        output_dir=output_dir,
        dx=DX,
        dy=DY,
        patch_size=patch_size,
        run_initial=run_initial
    )

    m0 = m.copy()
    run_initial = False

# M_output, final_score = dbs.direct_binary_search_decay(
#     M0=M0,
#     mx3_exe_path=MX3_EXE_PATH,
#     mx3_exe_convert_path=MX3_EXE_CONVERT_PATH,
#     template_path=TEMPLATE_PATH,
#     output_dir=OUTPUT_DIR,
#     initial_patch_size=3,
#     max_iterations=50,
#     tolerance=0.01
# )

# print(f"Final score: {final_score:.6g}")