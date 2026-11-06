# Relaxed Subset Simulation for Reliability Estimation

Python implementation of **Relaxed Subset Simulation (ReSuS)**. This code accompanies the paper:

> **Relaxed subset simulation for reliability estimation.**
> *Reliability Engineering & System Safety*, 264 (2025), 111302. doi: [10.1016/j.ress.2025.111302](https://doi.org/10.1016/j.ress.2025.111302)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Built-in examples and settings](#built-in-examples-and-settings)
- [Custom performance function](#custom-performance-function)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

## Overview

Relaxed subset simulation (ReSuS) estimates the failure probability

$$
P_f=\int_{\mathbb R^d}\mathbf1_{\{g(\mathbf u)\le0\}}\,\varphi_d(\mathbf u)\,\mathrm d\mathbf u.
$$

Here $\mathbf U\sim\mathcal N(\mathbf0,\mathbf I_d)$ is the standard-normal reference input, $\varphi_d$ its density, and $g$ the limit-state function; failure occurs when $g\le0$. ReSuS samples hybrid intermediate domains defined by LSF thresholds and Gaussian tails. After sampling, `ReSuSProduct` computes the original sequential probability-product estimate.

The default sampler is `ES`; `CW_aMH`, `aCS`, and `HMC` are also available.

## Requirements

- Python 3.10 or later and NumPy ≥1.23; tested with Python 3.12.14 / NumPy 2.3.5. See `requirements.txt`.
- Each repository runs independently, with no sibling-toolbox import or external example data.

## Repository structure

```text
├── LICENSE
├── Limit_state_functions/
│   ├── LSF_2D_ChangingTopological.py
│   ├── LSF_2D_LinearLog.py
│   ├── LSF_2D_PiecewiseLinear.py
│   └── __init__.py
├── MCMC_samplers/
│   ├── CW_aMH.py
│   ├── ES.py
│   ├── HMC.py
│   ├── MCMCBase.py
│   ├── __init__.py
│   └── aCS.py
├── Operations_on_Logarithmic_values/
│   ├── __init__.py
│   └── logGauss.py
├── README.md
├── ReSuS.py
├── SeMCS.py
├── Support_Fun/
│   ├── ReSuSProduct.py
│   └── __init__.py
├── drive_resus.py
└── requirements.txt
```

## Quick start

Set `MaxIte` to the maximum number of sampled levels. A scalar `NumSam` is reused; a vector is a finite per-level budget. The run returns when either limit is reached, and `flg_cvg` reports the method's own convergence criterion. Entries in `G` patch intermediate states; omitted fields and carried `lambda`/`HMC_time` are retained.

Run from this repository's root:

```bash
python -m pip install -r requirements.txt
python drive_resus.py --example A --n 1000 --seed 411 --sampler ES --profile paper
python drive_resus.py --example C --n 1000 --seed 411 --sampler aCS --profile paper
```

To change the problem, sample size, seed, sampler or profile, edit the driver settings or pass the corresponding arguments shown above.

## Built-in examples and settings

| Example | Model / dimension | Driver selector |
|---|---|---|
| A | Piecewise-linear LSF; 2 independent standard normals | `A` |
| B | Nonlinear changing-topology LSF; 2 independent standard normals | `B` |
| C | Linear / logistic LSF, paper Eq. (26); 2 independent standard normals | `C` |

The defaults are example A, 5000 samples per level, seed 1, conditional probability 0.1, `ES`, and the `legacy` profile. The maximum number of levels is 100. The quick-start commands select 1000 samples and the `paper` profile explicitly.

- `legacy` uses midpoint thresholds and strict seed eligibility.
- `paper` uses order-statistic thresholds, inclusive eligibility and Algorithm 1 chain-count rounding.
- Both profiles use the probability-product estimator.

`CW_aMH` uses seed-spread-scaled uniform increments in whitened Gaussian-reference coordinates. `ES` uses coordinatewise initial angles in `[0, pi]`. `aCS` starts with `sigma0=1` and `lambda0=0.6`, uses five adaptation groups and target acceptance 0.44, and carries lambda between levels. `HMC` uses analytic Gaussian-reference rotation and carries its adapted trajectory time between levels. Fixed options belong to the sampler; evolving values belong to the model's `G`.

## Custom performance function

Write a function with the same interface as the files in `Limit_state_functions/`: it accepts a NumPy array of shape `(dimension, sample_count)` in standard-normal reference coordinates and returns one finite LSF value per column, with failure defined by `g <= 0`.

Replace the selected LSF in `drive_resus.py` and set `model.Ndim`. For a separate script, construct `ReSuS`, set `lsf` and `Ndim`, bind the model to a sampler, construct `simulation = SeMCS(sampler)`, and call `ReSuSProduct(simulation.RunIte(rng))` as in the driver. For nonnormal physical inputs, include the reference-to-physical transformation in your LSF.

## Outputs

The raw sampling history `g` contains `Nite` sampled states followed by one final candidate state. `Nsze` records the chain and step counts at each sampled level.

The driver returns one result with the following fields:

| Field | Meaning |
|---|---|
| `pf`, `log_pf` | failure-probability estimate and its natural logarithm; zero estimates are retained as `0` and `-Inf` |
| `conditional_probabilities` | intermediate transition factors followed by the final true-failure factor |
| `log_normalizers` | cumulative log-normalizer estimates for the sampled levels |
| `samples`, `values` | sampled reference-space inputs and LSF values at each level |
| `sample_counts` | actual sample count at each level |
| `n_calls` | total LSF evaluations, including rejected proposals |
| `converged` | whether the method's stopping condition was reached |
| `estimator` | `product` |

If `converged` is false, the run ended before the stopping condition was reached. Postprocessing uses the actual sample counts.

## Citation

If you use this code in your research, please cite:

> Li, B., Xia, W., & Liao, Z. (2025). Relaxed subset simulation for reliability estimation. *Reliability Engineering & System Safety*, 264, 111302. https://doi.org/10.1016/j.ress.2025.111302

## License

This project is licensed under the [MIT License](LICENSE). Original copyright and source notices are retained. [Other language](https://github.com/ZihanLiao-ZJU/Relaxed-Subset-Simulation).
