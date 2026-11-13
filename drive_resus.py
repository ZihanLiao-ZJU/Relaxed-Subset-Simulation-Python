"""Run either of the three standard-normal numerical examples."""

from SeMCS import SeMCS
from ReSuS import ReSuS
from Support_Fun.ReSuSProduct import ReSuSProduct
from MCMC_samplers import CW_aMH, aCS, HMC, ES
from Limit_state_functions.LSF_2D_PiecewiseLinear import LSF_2D_PiecewiseLinear
from Limit_state_functions.LSF_2D_ChangingTopological import LSF_2D_ChangingTopological
from Limit_state_functions.LSF_2D_LinearLog import LSF_2D_LinearLog

# %% Performance function
model = ReSuS()
model.lsf = LSF_2D_PiecewiseLinear       # Example A
# model.lsf = LSF_2D_ChangingTopological  # Example B
# model.lsf = LSF_2D_LinearLog            # Example C
model.Ndim = 2

# %% Sampler and simulation
SamplerClass = ES                        # CW_aMH, aCS, HMC or ES
simulation = SeMCS(SamplerClass(model))
simulation.NumSam = 5000
raw = simulation.RunIte()

# %% Failure probability
result = ReSuSProduct(raw)
print(f"{model.lsf.__name__}: Pf={result['pf']:.8g}; LSF calls={result['n_calls']}; converged={result['converged']}")
