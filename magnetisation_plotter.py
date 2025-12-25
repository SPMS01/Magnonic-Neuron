import numpy as np
import matplotlib.pyplot as plt
import os
from utils import Dimension
from matplotlib.colors import SymLogNorm

def plot_magnetisation(dx: float, dy: float, file_path: str, output_path: str):
    data = np.load(file_path)[Dimension.X.value, 0]

    plt.figure(figsize=(10, 5))
    vabs = np.max(np.abs(data))
    
    # Calculate aspect ratio: physical height per cell / physical width per cell
    # 5nm / 20nm = 0.25
    cell_aspect = dy / dx

    plt.imshow(data, 
               cmap='seismic', 
               vmin=-vabs, 
               vmax=vabs, 
               origin='lower',
               aspect=cell_aspect) # This stretches the X-axis visually

    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]') 

    # plt.xlim(600, 1200)
    plt.xlabel('X (cell index)')
    plt.ylabel('Y (cell index)')
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    plot_magnetisation(dx=20e-9, dy=5e-9, file_path="./neuron.out/m_full001000.npy", output_path="updated pwease.svg")
    # plot_magnetisation(dx=20e-9, dy=10e-9, file_path="./coupler.out/m_full003000.npy", output_path="coupler_mx_100.svg")