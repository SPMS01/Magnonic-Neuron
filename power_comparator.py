from utils import DetectorRegion, DetectorRegionType
import detector
import os
import re
import numpy as np

def power_comparator(input_dir: str, 
                     p1: DetectorRegion, 
                     p2: DetectorRegion,
                     debug=False) -> float:
    frame_count = len([name for name in os.listdir(input_dir) if re.fullmatch(r"m\d+\.npy", name)])
    m1 = detector.detect_waves(x_range=p1.x_range,
                               y_range=p1.y_range,
                               input_dir=input_dir,
                               frame_count=frame_count,
                               dt=50e-12,
                               file_prefix="m",
                               detector_name="p1_detector",
                               debug=debug)
    m2 = detector.detect_waves(x_range=p2.x_range,
                               y_range=p2.y_range,
                               input_dir=input_dir,
                               frame_count=frame_count,
                               dt=50e-12,
                               file_prefix="m",
                               detector_name="p2_detector",
                               debug=debug)
    
    # take final 50% of the signal to compute power
    p1 = np.mean(m1[int(0.5 * frame_count):]**2)
    p2 = np.mean(m2[int(0.5 * frame_count):]**2)

    if debug:
        print(f"Power at p1: {p1}")
        print(f"Power at p2: {p2}")

    return p1 / (p1 + p2)
    
if __name__ == "__main__":
    # if we take the bottom waveguide as p1 and the top waveguide as p2

    INPUT_DIR = "new_coupler.out"
    region1 = DetectorRegion(DetectorRegionType.OTHER, x_range=(600, 601), y_range=(0, 5))
    region2 = DetectorRegion(DetectorRegionType.OTHER, x_range=(500, 501), y_range=(7, 12))
    ratio = power_comparator(INPUT_DIR, region1, region2, debug=True)

    print(f"Power ratio p1 / (p1 + p2): {ratio}")