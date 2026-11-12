"""Run either of the three standard-normal numerical examples."""

import numpy as np
from SeMCS import SeMCS
from ReSuS import ReSuS
from Support_Fun.ReSuSProduct import ReSuSProduct
from MCMC_samplers import CW_aMH, aCS, HMC, ES
from Limit_state_functions.LSF_2D_PiecewiseLinear import LSF_2D_PiecewiseLinear
from Limit_state_functions.LSF_2D_ChangingTopological import LSF_2D_ChangingTopological
from Limit_state_functions.LSF_2D_LinearLog import LSF_2D_LinearLog


# %% Problem and sampling settings
example = 'A'
Nsam = 5000
SamplerClass = ES           # CW_aMH, aCS, HMC or ES
p0 = 0.1
profile = 'legacy'

# %% Define problem and run simulation
examples = {'A': LSF_2D_PiecewiseLinear, 'B': LSF_2D_ChangingTopological, 'C': LSF_2D_LinearLog}
if example.upper() not in examples:
    raise ValueError('example must be A, B or C')
model = ReSuS(profile=profile)
model.lsf = examples[example.upper()]
model.Ndim = 2
kernel = SamplerClass(model)
kernel.BayStc = model
simulation = SeMCS(kernel)
simulation.NumSam = Nsam
simulation.ConPrb = p0
result = ReSuSProduct(simulation.RunIte(np.random.default_rng()))
print(
    f"Example {example.upper()}: Pf={result['pf']:.8g}; LSF calls={result['n_calls']}; converged={result['converged']}"
)
