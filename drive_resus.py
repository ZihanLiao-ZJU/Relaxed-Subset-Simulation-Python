"""Run either of the two standard-normal numerical examples."""
import argparse
import numpy as np
from SeMCS import SeMCS
from LSF_2D_PiecewiseLinear import LSF_2D_PiecewiseLinear
from LSF_2D_ChangingTopological import LSF_2D_ChangingTopological


def drive_resus(example='A', n=5000, seed=1):
    examples = {'A': LSF_2D_PiecewiseLinear, 'B': LSF_2D_ChangingTopological}
    if example.upper() not in examples:
        raise ValueError('example must be A or B')
    result = SeMCS(examples[example.upper()], 2, n).run(np.random.default_rng(seed))
    print(f"Example {example.upper()}: Pf={result['pf']:.8g}; LSF calls={result['n_calls']}; converged={result['converged']}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['A', 'B'], default='A')
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    drive_resus(args.example, args.n, args.seed)
