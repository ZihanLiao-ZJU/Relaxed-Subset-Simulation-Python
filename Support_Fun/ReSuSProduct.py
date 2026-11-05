"""Native ReSuS probability-product postprocessing."""
import numpy as np

def ReSuSProduct(g, log_z):
    final_probability = float(np.mean(g <= 0))
    log_pf = float(log_z + np.log(final_probability)) if final_probability > 0 else -np.inf
    pf = float(np.exp(log_pf))
    return pf, log_pf, final_probability
