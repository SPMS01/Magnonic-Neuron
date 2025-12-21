import numpy as np
import os
import time
import utils
import objective_function
import glob
import shutil

post_coupler_output_detector_region = utils.DetectorRegion(
    region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    x_range=(719, 720),
    y_range=(0, 15)
)

def get_patch_coordinates(M: np.ndarray, patch_size: int) -> list[tuple[int, int]]:
    """
    Get the top-left coordinates of all possible patches of given size in the matrix M.
    """
    h, w = M.shape
    return [(i, j) for i in range(0, h - patch_size + 1) for j in range(0, w - patch_size + 1)]

def flip_patch(M: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    M[y:y + patch_size, x:x + patch_size] = 1 - M[y:y + patch_size, x:x + patch_size]
    return M

def direct_binary_search_decay(M0: np.ndarray,
                         mx3_exe_path: str,
                         mx3_exe_convert_path: str,
                         template_path: str,
                         output_dir: str,
                         initial_patch_size: int = 1,
                         max_iterations: int = 100,
                         tolerance: float = 0.01) -> tuple[np.ndarray, float]:
    """
    Args:
        M0 (np.ndarray): Design matrix (2D array of 0s and 1s).
        mx3_exe_path (str): Path to the Mumax3 executable.
        mx3_exe_convert_path (str): Path to the Mumax3 convert executable.
        template_path (str): Path to the Mumax3 template file.
        output_dir (str): Directory to save output files.
        initial_patch_size (int): Initial size of the patches to flip.
        max_iterations (int): Maximum number of iterations to perform.
        tolerance (float): Minimum relative improvement in score to consider a flip successful.

    The principle of inverse design is that good improvements are kept until no further improvements can be made. Therefore, we converge at a relatively good design. 
    
    DBS is one way to converge onto this design. We start with a randomised grid of YIG and air. Then, we flip one cell at random at a time (YIG to air or vice versa) and check if we get a higher score. If a higher score is achieved, then such a flip is kept. All the cells are eventually flipped. This is considered one run. If there is any improvement in this one run, DBS is run again, flipping each cell in another random order. A mostly optimised structure is achieved when no more improvements are found.
    
    DBS-decay is an improved method that was found that performed better in inverse design. Rather than flipping one cell at a time, more cells are flipped at a time (n-by-n, n > 1) which can provide faster initial convergence. When DBS is done for n-by-n cells, DBS is done to (n-1)-by-(n-1) cells, until n = 1. This allows for initial faster convergence before the design is refined with smaller cells. 
    """
    M = M0.copy()
    best_score = 0
    
    with open(os.path.join(output_dir, "dbs_scores.csv"), "w+") as f:
        f.write("Iteration,Score,Flipped\n")
        patch_size = initial_patch_size

        for i in range(max_iterations): # loop for iteratons
            improved = False
            coords = get_patch_coordinates(M, patch_size=patch_size)
            np.random.shuffle(coords) # shuffle coordinates to randomise flip order

            run_folder = os.path.join(output_dir, f"iteration_{i:06d}")
            os.mkdir(run_folder)

            for j, (y, x) in enumerate(coords): # loop through each patch
                start = time.time()

                output_folder = os.path.join(run_folder, f"patch_{j:06d}")
                os.mkdir(output_folder)

                mx3_file_path = os.path.join(output_folder, f"{j:06d}_design.mx3")

                M = flip_patch(M, x, y, patch_size=1)
                # utils.generate_mx3_design(M, mx3_file_path, template_path, x_offset=150, y_offset=16)
                utils.generate_mx3_design(M, mx3_file_path, template_path, x_offset=150, y_offset=7)
                output = utils.run_mx3(mx3_exe_path, mx3_exe_convert_path, mx3_file_path, output_folder)
                mumax_output_folder = f"{output_folder}/{j:06d}_design.out"
                # score = objective_function.evaluate_objective(
                #     detector_regions=[post_coupler_output_detector_region],
                #     input_dir=mumax_output_folder,
                #     frame_count=len([name for name in os.listdir(mumax_output_folder) if name.endswith('.npy')]),
                #     dt=50e-12,
                #     debug=False
                # )
                score = objective_function.temporary_evaluate_objective(mumax_output_folder)
                flipped = False

                # check if improvement is significant (>1%)
                if score > 0 and (best_score == 0 or (score - best_score) / best_score > tolerance):
                    print(f"[Iteration {i}, Patch {j}] Improved score: {score:.6g} (best: {best_score:.6g}). Time taken: {time.time() - start:.2f} s; Difference: {score - best_score:.6g}; Patch size: {patch_size}")    

                    best_score = score
                    improved = True
                    flipped = True

                    # move debug plots to output folder (both jpg and png)
                    for file in glob.glob(f"{mumax_output_folder}/*.jpg"):
                        shutil.move(file, output_folder)
                    for file in glob.glob(f"{mumax_output_folder}/*.png"):
                        shutil.move(file, output_folder)
                    shutil.rmtree(mumax_output_folder)

                    # save M to output folder
                    np.save(os.path.join(output_folder, "M.npy"), M)
                else:
                    print(f"[Iteration {i}, Patch {j}] No improvement. Score: {score:.6g} (best: {best_score:.6g}). Time taken: {time.time() - start:.2f} s; Difference: {score - best_score:.6g}; Patch size: {patch_size}")
                    M = flip_patch(M, x, y, patch_size=patch_size) # flip back
                    shutil.rmtree(output_folder) # remove folder to save space

                f.write(f"{i},{best_score},{flipped}\n")

            if not improved:
                if patch_size > 1:
                    patch_size -= 1
                else:
                    print("No further improvement possible. Terminating.")
                    break

    return M, best_score