import numpy as np
import matplotlib.pyplot as plt
import os
from utils import Dimension
from matplotlib.colors import SymLogNorm

def plot_magnetisation(file_path: str, output_path: str):
    data = np.load(file_path)[Dimension.X.value, 0]

    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    # plt.imshow(data, cmap='jet', origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

    plt.xlim(300, 1000) # default: (600, 850)
    plt.xlabel('X index')
    plt.ylabel('Y index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_magnetisation("./coupler_shenanigans.out/m003000.npy", "baller.svg")