import utils
import detector
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import utils
import time
import matplotlib
from magnetisation_plotter import plot_magnetisation
import re
from power_comparator import power_comparator

def calculate_switch_functionality(mx3_output_dir: str,
                                   output_dir: str,
                                   template_path: str,
                                   p1: utils.DetectorRegion,
                                   p2: utils.DetectorRegion,
                                   f0: float,
                                   dt: float = 50e-12,
                                   debug: bool = True):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    power_ratios = []
    for i in range(3, 63, 3):
        t0 = time.time()

        line = f"B_ext.SetRegion(excitation_region, Vector({i}e-3 * sin(2 * pi * {f0} * t), 0, 300e-3))"

        with open(template_path, "r") as template_file:
            content = template_file.read()
            content = content.replace("// {{ INSERT EXCITATION FIELD HERE }}", line)
        
        mx3_file_path = os.path.join(output_dir, f'switch_functionality_{f0/1e9:.2f}GHz_{i}mT.mx3')
        with open(mx3_file_path, "w+") as mx3_file:
            mx3_file.write(content)

        utils.run_mx3("mumax3", "mumax3-convert", mx3_file_path, output_dir)

        mx3_output_dir = os.path.join(output_dir, f'switch_functionality_{f0/1e9:.2f}GHz_{i}mT.out')
        print(mx3_output_dir)

        ratio = power_comparator(mx3_output_dir, p1, p2, debug=debug)
        print(f"Power ratio p1 / (p1 + p2): {ratio}")
        power_ratios.append((i, ratio))
        
        with open(os.path.join(output_dir, 'power_ratios.csv'), 'a+') as f:
            f.write(f"{i},{ratio}\n")

        print(f"Completed simulation for B_ext = {i} mT. Time elapsed: {time.time() - t0:.2f} seconds")

        plot_magnetisation(dx=20e-9, dy=20e-9, file_path=os.path.join(mx3_output_dir, "m003000.npy"), output_path=os.path.join(output_dir, f"{i}mT_magnetisation_plot.svg"))
        shutil.rmtree(mx3_output_dir)

    fields, ratios = zip(*power_ratios)
    return fields, ratios

if __name__ == "__main__":
    # r1 is in the bottom waveguide
    # r2 is in the top waveguide

    F0 = 6.7e9  # Target frequency in Hz
    OUTPUT_DIR = f"./coupler_functionality_results_50nm_separation{F0*1e-9:.2f}GHz"
    TEMPLATE_PATH = "coupler.mx3"

    # r1 = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.PRE_COUPLER_OUTPUT,
    #     x_range=(600, 601),
    #     y_range=(0, 20)
    # )

    # r2 = utils.DetectorRegion(
    #     region_type=utils.DetectorRegionType.POST_COUPLER_OUTPUT,
    #     x_range=(450, 451),
    #     y_range=(15, 25)
    # )
    
    # fields, ratios = calculate_switch_functionality(
    #     mx3_output_dir="./coupler.out",
    #     output_dir=OUTPUT_DIR,
    #     template_path=TEMPLATE_PATH,
    #     p1=r1,
    #     p2=r2,
    #     f0=F0,
    #     dt=50e-12,
    #     debug=False
    # )

    # plot the results
    with open(os.path.join(OUTPUT_DIR, 'power_ratios.csv'), 'r') as f:
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

    plt.figure()
    ax = plt.gca()

    # Plot
    ax.plot(fields, np.array(ratios) * 100, marker='o', color='black')
    ax.set_xlabel(r"$b_0 \ \mathrm{(mT)}$", fontsize=20)
    ax.set_ylabel(r"$P_{\mathrm{1}} \ / (\ P_{\mathrm{1}} \mathrm{+} P_{\mathrm{2}}) \ \mathrm{(\%)}$", fontsize=20)
    ax.tick_params(labelsize=16, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # ax.text(
    #     0.02, 0.98,
    #     r"$\mathrm{d)}$",
    #     transform=ax.transAxes,
    #     fontsize=38,
    #     va="top",
    #     ha="left"
    # )

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, 'switch_functionality_plot.svg'),
        dpi=300
    )
    plt.close()