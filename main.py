from detector import detect_waves
from utils import Dimension
import numpy as np

M0 = np.random.choice([0, 1], size=(50, 50))

detect_waves(
    x_range=(130, 131),
    y_range=(0, 15),
    input_dir='./ring-resonator.out',
    frame_count=2001,
    dt=50e-12,
    dimension=Dimension.Y
)