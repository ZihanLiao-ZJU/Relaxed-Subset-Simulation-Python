import numpy as np


def LSF_2D_PiecewiseLinear(x):
    a = np.where(x[0] > 3.5, 4 - x[0], 0.85 - 0.1 * x[0])
    b = np.where(x[1] > 2, 0.5 - 0.1 * x[1], 2.3 - x[1])
    return np.minimum(a, b)
