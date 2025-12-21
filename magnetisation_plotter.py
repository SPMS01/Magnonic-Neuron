import numpy as np
import matplotlib.pyplot as plt
import os
from utils import Dimension
from matplotlib.colors import SymLogNorm

# def plot_magnetisation(file_path: str, output_path: str):
#     data = np.load(file_path)[Dimension.X.value, 0]

#     vabs = np.max(np.abs(data))

#     plt.figure(figsize=(10, 4))
#     plt.imshow(
#         data,
#         cmap="seismic",
#         origin="lower",
#         norm=SymLogNorm(
#             linthresh=vabs * 1e-3,   # linear region around 0
#             vmin=-vabs,
#             vmax=vabs,
#             base=10
#         )
#     )

#     plt.colorbar(label="Magnetisation (arb. units, symlog)")
#     plt.xlabel("X index")
#     plt.ylabel("Y index")
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()


def plot_magnetisation(file_path: str, output_path: str):
    data = np.load(file_path)[Dimension.X.value, 0]

    plt.figure(figsize=(10, 4))
    vabs = np.max(np.abs(data))
    plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
    # plt.imshow(data, cmap='jet', origin='lower')
    plt.colorbar(label='Magnetisation (arb. units)')
    plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

    # plt.xlim(400, 1000) # default: (600, 850)
    plt.xlabel('X index')
    plt.ylabel('Y index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    f = 7.5 # in GHz
    m = 1 # in mT
    # plot_magnetisation(f"./paper_coupler_{f}GHz_{m}mT.out/m_full003000.npy", f"paper_coupler_{f}GHz_{m}mT.svg")
    plot_magnetisation("./new_coupler.out/m_full003000.npy", "resonator.svg")

# FILE_PREFIX = "m"
# INPUT_DIR = './ring-resonator.out'
# DT = 50e-12
# FRAME_COUNT = 1001
# DIMENSION = Dimension.X

# x_start, x_end = 224, 225
# y_start, y_end = 35, 50

# # === LOAD DATA ===
# data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}000800.npy'))[Dimension.X.value, 0]