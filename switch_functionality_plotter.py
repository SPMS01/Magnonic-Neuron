import utils
import detector
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import utils
import time
import matplotlib

def calculate_switch_functionality(input_region: utils.DetectorRegion,
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

def plot_detector_regions(input_region: utils.DetectorRegion,
                          output_region: utils.DetectorRegion,
                          input_dir: str = "./coupler.out"):
    start_detector_start = (input_region.x_range[0], input_region.y_range[0])
    start_detector_end = (input_region.x_range[1], input_region.y_range[1])

    end_detector_start = (output_region.x_range[0], output_region.y_range[0])
    end_detector_end = (output_region.x_range[1], output_region.y_range[1])

    data = np.load(os.path.join(input_dir, f'm001000.npy'))[utils.Dimension.X.value, 0]
    
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
    plt.savefig('switch_functionality_detector_regions.png', dpi=300)

if __name__ == "__main__":
    OUTPUT_DIR = "./coupler_functionality_results"
    TEMPLATE_PATH = "coupler_switch_functionality_template.mx3"

    # for neuron switch functionality
    # pre_neuron_input_region = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
    #     x_range=(200, 201),
    #     y_range=(67, 82)
    # )

    # post_neuron_output_region = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    #     x_range=(650, 651),
    #     y_range=(0, 15)
    # )

    # for coupler switch functionality
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
    
    # fields, ratios = calculate_switch_functionality(
    #     input_region=pre_coupler_input_region,
    #     output_region=post_coupler_output_region,
    #     output_dir=OUTPUT_DIR,
    #     template_path=TEMPLATE_PATH,
    #     dt=50e-12,
    #     debug=True
    # )

    # plot the results
    with open(os.path.join(OUTPUT_DIR, 'transmission_ratios.csv'), 'r') as f:
        lines = f.readlines()
        fields = []
        ratios = []
        for line in lines:
            field, ratio = line.strip().split(',')
            fields.append(float(field))
            ratios.append(float(ratio))

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)

    plt.figure()
    plt.plot(fields, np.array(ratios) * 100, marker='o', color='black')
    plt.xlabel(r"$b_0 \ \mathrm{(mT)}$", fontsize=20)
    plt.ylabel(r"$P_{\mathrm{output}} \ / \ P_{\mathrm{input}} \ \mathrm{(\%)}$", fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'switch_functionality_plot.svg'), dpi=300)

    # plot_detector_regions(
    #     input_region=pre_coupler_input_region,
    #     output_region=post_coupler_output_region,
    #     input_dir="./coupler.out"
    # )