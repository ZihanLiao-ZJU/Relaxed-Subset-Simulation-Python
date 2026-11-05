"""Standard-normal log density used by the original ReSuS implementation."""
import numpy as np

def logGauss(x, dimension):
    return -.5*np.sum(x*x, axis=0)-dimension/2*np.log(2*np.pi)
