"""Example C, Eq. (26), DOI: 10.1016/j.ress.2025.111302.
Independent standard-normal inputs; samples are columns.
The paper uses 5 - x1 without the 0.03 scale in the earlier supplied code.
"""
import numpy as np


def LSF_2D_LinearLog(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError("Example C requires a (2, N) input matrix")
    with np.errstate(over="ignore"):
        logistic = 1.0 / (1.0 + np.exp(-2.0 * (x[1] + 4.0)))
    return np.minimum(5.0 - x[0], logistic - 0.5)
