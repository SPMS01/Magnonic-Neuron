import numpy as np
import matplotlib.pyplot as plt
import os
from utils import Dimension
from matplotlib.colors import SymLogNorm
import matplotlib

def plot_magnetisation(dx: float, dy: float, file_path: str, output_path: str):
    data = np.load(file_path)[Dimension.X.value, 0]

    plt.figure(figsize=(10, 5))
    vabs = np.max(np.abs(data))
    
    # Calculate aspect ratio: physical height per cell / physical width per cell
    # 5nm / 20nm = 0.25
    cell_aspect = dy / dx

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    # no xtics/ytics and no frame
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    plt.imshow(data, 
               cmap='seismic', 
               vmin=-vabs, 
               vmax=vabs, 
               origin='lower',
               aspect=cell_aspect) # This stretches the X-axis visually

    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]') 
    # plt.title("39 mT", fontsize=28)

    plt.xlim(600, 1200)
    plt.xlabel('X (cell index)')
    plt.ylabel('Y (cell index)')
    plt.tight_layout()
    plt.savefig(output_path)
    # # plt.show()
    plt.close()
    

if __name__ == "__main__":
    fp = "000202_design.out"
    # plot_magnetisation(dx=20e-9, dy=5e-9, file_path="single_waveguide_6.7GHz_33mT.out/m_full003000.npy", output_path="single_waveguide_6.7GHz_33mT.svg")
    # plot_magnetisation(dx=20e-9, dy=5e-9, file_path=fp+"/m_full003000.npy", output_path=fp+".svg")
    plot_magnetisation(dx=20e-9, dy=10e-9, file_path="test_coupler_pair_100nm_gap.out/m003000.npy", output_path="new 100nm innovation coupler.svg")