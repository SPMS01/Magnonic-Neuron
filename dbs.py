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
    M = M0.copy()
    best_score = 0
    
    with open(os.path.join(output_dir, "dbs_scores.csv"), "w+") as f:
        f.write("Iteration,Score,Flipped\n")
        patch_size = initial_patch_size

        for i in range(max_iterations):
            improved = False
            coords = get_patch_coordinates(M, patch_size=patch_size)
            np.random.shuffle(coords)

            run_folder = os.path.join(output_dir, f"iteration_{i:06d}")
            os.mkdir(run_folder)

            for j, (y, x) in enumerate(coords):
                start = time.time()

                output_folder = os.path.join(run_folder, f"patch_{j:06d}")
                os.mkdir(output_folder)
                mx3_file_path = os.path.join(output_folder, f"{j:06d}_design.mx3")

                M = flip_patch(M, x, y, patch_size=1)
                utils.generate_mx3_design(M, mx3_file_path, template_path, x_offset=150, y_offset=16)
                output = utils.run_mx3(mx3_exe_path, mx3_exe_convert_path, mx3_file_path, output_folder)
                mumax_output_folder = f"{output_folder}/{j:06d}_design.out"
                score = objective_function.evaluate_objective(
                    detector_regions=[post_coupler_output_detector_region],
                    input_dir=mumax_output_folder,
                    frame_count=len([name for name in os.listdir(mumax_output_folder) if name.endswith('.npy')]),
                    dt=50e-12,
                    debug=False
                )
                flipped = False

                if score > 0 and (best_score == 0 or (score - best_score) / best_score > tolerance):
                    print(f"[Iteration {i}, Patch {j}] Improved score: {score:.6g} (best: {best_score:.6g}). Time taken: {time.time() - start:.2f} s; Difference: {score - best_score:.6g}; Patch size: {patch_size}")    

                    best_score = score
                    improved = True
                    flipped = True

                    for file in glob.glob(f"{mumax_output_folder}/*.jpg"):
                        shutil.move(file, output_folder)
                    shutil.rmtree(mumax_output_folder)

                    # save M to output folder
                    np.save(os.path.join(output_folder, "M.npy"), M)
                else:
                    print(f"[Iteration {i}, Patch {j}] No improvement. Score: {score:.6g} (best: {best_score:.6g}). Time taken: {time.time() - start:.2f} s; Difference: {score - best_score:.6g}; Patch size: {patch_size}")
                    M = flip_patch(M, x, y, patch_size=patch_size)
                    shutil.rmtree(output_folder)

                f.write(f"{i},{best_score},{flipped}\n")

            if not improved:
                if patch_size > 1:
                    patch_size -= 1
                else:
                    print("No further improvement possible. Terminating.")
                    break

    return M, best_score