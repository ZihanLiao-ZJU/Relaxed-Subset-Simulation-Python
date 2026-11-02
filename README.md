# ReSuS — Python

## Overview
Relaxed subset simulation estimates failure probabilities with hybrid LSF / Gaussian-tail intermediate domains. This revision restores the **original sequential probability-product estimator**, Eqs. (21)–(24) of the cited paper, in place of the previous historical-sample MIS estimator. The default sampler remains `ES`; the four unified kernels remain unchanged.

## Structure
- `drive_resus.py` — example selection, settings and execution.
- `SeMCS.py` — iteration, stored levels and probability-product estimation.
- `ReSuS.py` — hybrid distributions, thresholds, seed selection and convergence.
- `LSF_2D_PiecewiseLinear.py`, `LSF_2D_ChangingTopological.py`, `LSF_2D_LinearLog.py` — examples A–C.
- `MCMCBase.py`, `CW_aMH.py`, `aCS.py`, `HMC.py`, `ES.py` — locally included sampling interface and kernels.

## Quick start
Run from this repository's root:

```bash
python -m pip install -r requirements.txt
python drive_resus.py --example A --n 1000 --seed 411 --sampler ES --profile paper
python drive_resus.py --example C --n 1000 --seed 411 --sampler aCS --profile paper
```
The returned result includes `pf`, `log_pf`, `conditional_probabilities`, actual `n_calls`, `converged`, sampled levels and `sampler_history`. `converged = false` is an unfinished run, not a validated estimate. `estimator` identifies this revision as `product`.

## Examples and samplers
| Example | Model / dimension | Driver selector |
|---|---|---|
| A | Piecewise-linear LSF; 2 independent standard normals | `A` |
| B | Nonlinear changing-topology LSF; 2 independent standard normals | `B` |
| C | Linear / logistic LSF, paper Eq. (26); 2 independent standard normals | `C` |

- Available samplers: `CW_aMH`, `aCS`, `HMC`, `ES`; default `ES`. The quick start shows explicit selection.
- Defaults are unchanged: example A, N=5000, seed 1, p0=0.1 and `legacy` profile. Explicit `paper` runs above use N=1000; the level budget defaults to 100.
- `legacy` retains midpoint thresholds and strict seed eligibility. `paper` uses order-statistic thresholds, inclusive eligibility and Algorithm 1 chain-count rounding. **Both profiles now use the product estimator**; selecting `paper` does not replace the unified ES with a historical ESS kernel.
- C uses the paper's unscaled `5-x1` branch, not the `0.03` scale present in an earlier supplied file. No D–H models are bundled.
- The `SeMCS` constructor accepts a model and its dimension. Models return one finite LSF value per column of a dimension-by-sample input. Custom physical/nonnormal models must be mapped to Gaussian/Nataf reference coordinates; common kernels are not restricted to 2D.

## Dependencies
- Python 3.10 or later and NumPy ≥1.23; tested with Python 3.12.14 / NumPy 2.3.5. See `requirements.txt`.
- Each repository runs independently, with no sibling-toolbox import or external example data.

## Method & validation notes
- **Implementation:** each transition contributes its empirical hybrid-event probability on the **current sampled level**, divided by the actual number of samples (which can differ from requested N). The last factor is the true-failure proportion `g <= 0`; no extra unsampled hybrid factor is included. Factors are accumulated in log space. A zero estimate is retained as `pf=0`, `log_pf=-Inf`.
- **Sampler policy:** `CW_aMH` uses seed-spread-scaled **uniform random-walk increments in whitened Gaussian-reference coordinates**. `ES` is the coordinatewise initial-angle `[0, pi]` version, not a claim of identical historical scalar-angle ESS. `aCS` starts with `sigma0=1`, `lambda0=0.6` and carries the final lambda between levels (five adaptation groups, target acceptance 0.44). `HMC` uses analytic Gaussian-reference rotation; its adaptive trajectory time is carried between levels. It is not generic physical-coordinate gradient HMC.
- **Shared interface:** `SamGen(seed_u, seed_y, Nsteps)` returns `(u, y, Ncal, BayStc)` (MATLAB uses multiple outputs); arrays are dimension/function × chain × step. `G` carries thresholds and adaptation. Local copies of common kernels keep each repository independently runnable; no sibling repository is required.
- **Outputs:** `conditional_probabilities` contains the factors, `log_normalizers` their cumulative level normalizers, and `sample_counts` actual level sizes. `n_calls` counts all LSF evaluations, including rejected proposals. Standard deviation across independent runs is not the standard error of their mean. Random seeds do not identify equal streams across languages.
- **Validation scope:** product factors, terminal handling, count accounting and unchanged sampling trajectories were independently checked. A–C × ES/aCS/HMC completed 100-repeat main batches and separate 1000-repeat batches in both languages: 19,800 attempts, all converged. Additional four-kernel clean-entry and 3D/7D interface checks passed. The original paper did not report every optional-kernel combination.
- **Limitations:** implementation/accounting checks passed, but **not all paper statistics matched**. In particular, example C's empirical variability remains different, and the 1000-repeat mean estimates exceeded the analytic probability implied by the printed model; the cause is not established. Lower CoV is not evidence of exact reproduction. Historical MIS results do not validate this product revision. Paper batch-means variance bounds and p0 sensitivity grids are not reproduced here; bootstrap intervals are not substitutes for those bounds.

## Reference & license
Binbin Li, Weili Xia and Zihan Liao. **Relaxed subset simulation for reliability estimation.** *Reliability Engineering & System Safety* 264 (2025), 111302. [DOI](https://doi.org/10.1016/j.ress.2025.111302).

Licensed under the [MIT License](LICENSE). Original copyright and source notices are retained. [Other language](https://github.com/ZihanLiao-ZJU/Relaxed-Subset-Simulation).
