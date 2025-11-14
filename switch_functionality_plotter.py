import utils
import detector
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import utils
import time

def switch_functionality_plotter(input_region: utils.DetectorRegion,
                                 output_region: utils.DetectorRegion,
                                 output_dir: str,
                                 template_path: str,
                                 dt: float = 50e-12,
                                 debug: bool = True):
    start = time.time()

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    transmission_ratios = []

    for i in range(3, 63, 3):
        line = f'B_ext.SetRegion(excitation_region, Vector(0, {i}e-3 * sin(2 * pi * 1.5e9 * t), 200e-3))'

        with open(template_path, 'r') as template_file:
            content = template_file.read()
            content = content.replace("// {{ INSERT EXCITATION FIELD HERE }}", line)

        mx3_file_path = os.path.join(output_dir, f'switch_functionality_Bext_{i}mT.mx3')
        with open(mx3_file_path, 'w+') as mx3_file:
            mx3_file.write(content)

        utils.run_mx3("mumax3", "mumax3-convert", mx3_file_path, output_dir)

        input_dir = os.path.join(output_dir, f'switch_functionality_Bext_{i}mT.out')
        frame_count = len([name for name in os.listdir(input_dir) if name.endswith('.npy')])
        print(input_dir)

        region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

        for detector_region in [input_region, output_region]:
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

        Mx_pre = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 0]
        My_pre = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 1]
        power_pre = Mx_pre**2 + My_pre**2
        print(f"Average pre-coupler power: {np.mean(power_pre):.6g} a.u.")

        Mx_post = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
        My_post = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
        power_post = Mx_post**2 + My_post**2
        print(f"Average post-coupler power: {np.mean(power_post):.6g} a.u.")

        transmission_ratio = np.mean(power_post) / np.mean(power_pre)
        print(f"Transmission Percentage (Post/Pre): {transmission_ratio*100:.6g}%")

        transmission_ratios.append((i, transmission_ratio))

        with open(os.path.join(output_dir, 'transmission_ratios.csv'), 'a+') as f:
            f.write(f"{i},{transmission_ratio}\n")

        print(f"Completed simulation for B_ext = {i} mT. Time elapsed: {time.time() - start:.2f} seconds")

        shutil.rmtree(input_dir)

    end = time.time()
    print(f"Total simulation and processing time: {end - start:.2f} seconds")

    fields, ratios = zip(*transmission_ratios)
    return fields, ratios

def run_coupler_switch_functionality_plotter():
    pre_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
        x_range=(149, 150),
        y_range=(16, 31)
    )

    post_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
        x_range=(550, 551),
        y_range=(16, 31)
    )

    region_data: dict[utils.DetectorRegionType, np.ndarray] = {}

    for detector_region in [pre_coupler_output_detector_region, post_coupler_output_detector_region]:
        x_range = detector_region.x_range
        y_range = detector_region.y_range

        wave_data = detector.detect_waves(
            x_range=x_range,
            y_range=y_range,
            input_dir="./coupler.out", 
            frame_count=len([name for name in os.listdir("./coupler.out") if name.endswith('.npy')]),
            dt=50e-12,
            debug=True
        )

        region_data[detector_region.region_type] = wave_data

    Mx_pre = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 0]
    My_pre = region_data[utils.DetectorRegionType.PRE_COUPLER_OUTPUT][:, 1]
    power_pre = Mx_pre**2 + My_pre**2
    print(f"Average pre-coupler power: {np.mean(power_pre):.6g} a.u.")

    Mx_post = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 0]
    My_post = region_data[utils.DetectorRegionType.POST_COUPLER_OUTPUT][:, 1]
    power_post = Mx_post**2 + My_post**2
    print(f"Average post-coupler power: {np.mean(power_post):.6g} a.u.")

    transmission_ratio = np.mean(power_post) / np.mean(power_pre)
    print(f"Transmission Percentage (Post/Pre): {transmission_ratio*100:.6g}%")

    start_detector_start = (149, 16)
    start_detector_end = (150, 31)

    end_detector_start = (550, 16)
    end_detector_end = (551, 31)

    data = np.load(os.path.join('./coupler.out', f'm000400.npy'))[utils.Dimension.X.value, 0]

    # === PLOT ===
    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][utils.Dimension.X.value]}]')

    # Draw detector boxes
    plt.axvline(start_detector_start[0], color='cyan', linestyle='--')
    plt.axvline(start_detector_end[0], color='cyan', linestyle='--')
    plt.axhline(start_detector_start[1], color='cyan', linestyle='--')
    plt.axhline(start_detector_end[1], color='cyan', linestyle='--')
    plt.text(start_detector_start[0], start_detector_end[1]+1, 'Pre-coupler Detector Region', color='cyan')

    plt.axvline(end_detector_start[0], color='lime', linestyle='--')
    plt.axvline(end_detector_end[0], color='lime', linestyle='--')
    plt.axhline(end_detector_start[1], color='lime', linestyle='--')
    plt.axhline(end_detector_end[1], color='lime', linestyle='--')
    plt.text(end_detector_start[0], end_detector_end[1]+1, 'Post-coupler Detector Region', color='lime')

    plt.xlabel('X index')
    plt.ylabel('Y index')
    plt.tight_layout()
    plt.show()

    pass

if __name__ == "__main__":
    OUTPUT_DIR = "./switch_functionality_results"

    pre_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
        x_range=(149, 150),
        y_range=(16, 31)
    )

    post_coupler_output_detector_region = utils.DetectorRegion(
        region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
        x_range=(550, 551),
        y_range=(16, 31)
    )

    # pre_coupler_output_detector_region = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
    #     x_range=(149, 150),
    #     y_range=(0, 15)
    # )

    # post_coupler_output_detector_region = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    #     x_range=(550, 551),
    #     y_range=(0, 15)
    # )

    fields, ratios = switch_functionality_plotter(
        input_region=pre_coupler_output_detector_region,
        output_region=post_coupler_output_detector_region,
        output_dir=OUTPUT_DIR,
        template_path='./coupler_switch_functionality_template.mx3',
        dt=50e-12,
        debug=False
    )

    plt.figure(figsize=(8, 6))
    plt.plot(fields, ratios, marker='o')
    plt.title('Switch Functionality: Transmission Ratio vs External Magnetic Field')
    plt.xlabel('External Magnetic Field B_ext (mT)')
    plt.ylabel('Transmission Ratio (Post-Coupler / Pre-Coupler)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'switch_functionality_plot.png'))