import numpy as np
import matplotlib.pyplot as plt
from utils import generate_mx3_design

M = np.load('M.npy')
NEURON_TEMPLATE = "neuron_dbs_template.mx3"

generate_mx3_design(
    M=M,
    output_path="./neuron_generated.mx3",
    template_path=NEURON_TEMPLATE,
    x_offset=100,
    y_offset=21,
    height=200
)