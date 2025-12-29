from detector import detect_waves
import objective_function
import utils
import numpy as np
import os
import dbs
import shutil

def circle_layout():
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

    return ((dist <= outer_r) & (dist >= inner_r)).astype(np.uint8)

def random_layout():
    Ny, Nx = 200, 50

    # block size in CELLS
    bx_cells = 5     # 5 × 20 nm = 100 nm
    by_cells = 20    # 20 × 5 nm = 100 nm

    M = np.zeros((Ny, Nx), dtype=np.uint8)

    for by in range(0, Ny, by_cells):
        for bx in range(0, Nx, bx_cells):
            val = np.random.randint(0, 2)
            M[by:by+by_cells, bx:bx+bx_cells] = val

    return M


# Initial random design of YIG and air
# M0 = np.random.choice([0, 1], size=(50, 50))

M0 = circle_layout()

MX3_EXE_PATH = 'mumax3'
MX3_EXE_CONVERT_PATH = 'mumax3-convert'
OUTPUT_DIR = './dbs_output'
TEMPLATE_PATH = 'reduced_neuron_dbs_template.mx3'
FLIP_LIST_PATH = './flip_list.npy'
DX = 20e-9
DY = 5e-9
# PATCH_SIZE = 100e-9

# Clean output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

m0 = M0.copy()

for patch_size in [100e-9, 40e-9, 20e-9]:
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
        flip_list_path=FLIP_LIST_PATH,
        patch_size=patch_size
    )

    m0 = m.copy()

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