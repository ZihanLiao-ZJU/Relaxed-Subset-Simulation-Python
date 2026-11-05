"""Run either of the three standard-normal numerical examples."""

import argparse
import numpy as np
from SeMCS import SeMCS
from ReSuS import ReSuS
from Support_Fun.ReSuSProduct import ReSuSProduct
from MCMC_samplers import CW_aMH, aCS, HMC, ES
from Limit_state_functions.LSF_2D_PiecewiseLinear import LSF_2D_PiecewiseLinear
from Limit_state_functions.LSF_2D_ChangingTopological import LSF_2D_ChangingTopological
from Limit_state_functions.LSF_2D_LinearLog import LSF_2D_LinearLog


def drive_resus(example='A', n=5000, seed=1, sampler='ES', p0=0.1, profile='legacy'):
    examples = {'A': LSF_2D_PiecewiseLinear, 'B': LSF_2D_ChangingTopological, 'C': LSF_2D_LinearLog}
    if example.upper() not in examples:
        raise ValueError('example must be A, B or C')
    model = ReSuS(profile=profile)
    model.lsf = examples[example.upper()]
    model.Ndim = 2
    factories = {"CW_aMH": CW_aMH, "aCS": aCS, "HMC": HMC, "ES": ES}
    kernel = factories[sampler](model) if isinstance(sampler, str) else sampler
    kernel.BayStc = model
    simulation = SeMCS(kernel)
    simulation.NumSam = n
    simulation.ConPrb = p0
    result = ReSuSProduct(simulation.RunIte(np.random.default_rng(seed)))
    print(
        f"Example {example.upper()}: Pf={result['pf']:.8g}; LSF calls={result['n_calls']}; converged={result['converged']}"
    )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--sampler", choices=["CW_aMH", "ES", "aCS", "HMC"], default="ES")
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--profile", choices=["legacy", "paper"], default="legacy")
    args = parser.parse_args()
    drive_resus(args.example, args.n, args.seed, args.sampler, args.p0, args.profile)
