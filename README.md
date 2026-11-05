# ReSuS — Python

## Overview
Relaxed subset simulation (ReSuS) estimates small failure probabilities using hybrid LSF / Gaussian-tail intermediate domains and the original sequential probability-product estimator. The default sampler is `ES`; `CW_aMH`, `aCS`, and `HMC` are also available.

## Failure-probability integral and estimator

Let $\mathbf U\sim\mathcal N(\mathbf0,\mathbf I_d)$, with standard-normal density $\varphi_d$, and let $g(\mathbf u)$ be the limit-state function in these reference coordinates. The failure domain is $\mathcal F=\{\mathbf u:g(\mathbf u)\le0\}$. The failure probability is

$$
P_f=\int_{\mathbb R^d}\mathbf1_{\mathcal F}(\mathbf u)\,\varphi_d(\mathbf u)\,\mathrm d\mathbf u.
$$

ReSuS uses hybrid intermediate domains and their importance sampling densities:

$$
\mathcal A_0=\mathbb R^d,\qquad
\mathcal A_i=\{g(\mathbf u)\le b_i\}\cup\{\log\varphi_d(\mathbf u)\le a_i\},
$$

$$
q_i(\mathbf u)=\frac{\varphi_d(\mathbf u)\,\mathbf1_{\mathcal A_i}(\mathbf u)}{Z_i},\qquad
Z_i=\int_{\mathbb R^d}\mathbf1_{\mathcal A_i}(\mathbf u)\,\varphi_d(\mathbf u)\,\mathrm d\mathbf u,\qquad Z_0=1.
$$

Here $b_i$ is the LSF threshold and $a_i$ the Gaussian log-density threshold. The probability-product derivation requires nonincreasing thresholds (paper Eq. (20)). Let $m$ denote the last actually sampled level, with $b_m>0$. These conditions give $\mathcal A_0\supseteq\cdots\supseteq\mathcal A_m\supseteq\mathcal F$. The code chooses thresholds by empirical quantiles; it does not explicitly clamp each new threshold to the preceding value.

For this nested construction, define the transition probability $r_i$ and final failure proportion $r_F$ by

$$
r_i=\mathbb E_{q_{i-1}}[\mathbf1_{\mathcal A_i}]=\frac{Z_i}{Z_{i-1}},\qquad
r_F=\mathbb E_{q_m}[\mathbf1_{\mathcal F}].
$$

Then the failure integral becomes the native probability product (paper Eqs. (21)–(24)):

$$
P_f=Z_m\int_{\mathbb R^d}\mathbf1_{\mathcal F}(\mathbf u)q_m(\mathbf u)\,\mathrm d\mathbf u
=\left(\prod_{i=1}^{m}r_i\right)r_F.
$$

With $N_i$ actual samples $\mathbf u_k^{(i)}$ from level $i$, the implemented estimate is

$$
\widehat r_i=\frac1{N_{i-1}}\sum_{k=1}^{N_{i-1}}\mathbf1_{\mathcal A_i}(\mathbf u_k^{(i-1)}),\qquad
\widehat r_F=\frac1{N_m}\sum_{k=1}^{N_m}\mathbf1_{\mathcal F}(\mathbf u_k^{(m)}),
$$

$$
\widehat P_f=\left(\prod_{i=1}^{m}\widehat r_i\right)\widehat r_F,\qquad
\log\widehat P_f=\sum_{i=1}^{m}\log\widehat r_i+\log\widehat r_F.
$$

The empty product is 1 when $m=0$. At normal termination, the *next* LSF threshold has reached zero or below; the final factor is evaluated on the last **actually sampled** domain, whose LSF threshold is still positive. No additional unsampled hybrid-event factor is included. `conditional_probabilities` stores the transition factors followed by the true-failure factor; `log_normalizers` stores cumulative log-normalizer estimates. `ReSuSProduct` performs this native postprocessing rather than balance-heuristic MIS. All averages use actual level sizes, and MCMC correlation and adaptive thresholds remain relevant to uncertainty.

## Structure
- `drive_resus.py` — example selection, settings and execution.
- `SeMCS.py` — iteration, stored levels and probability-product estimation.
- `ReSuS.py` — hybrid distributions, thresholds, seed selection and convergence.
- `Limit_state_functions/LSF_2D_PiecewiseLinear.py`, `Limit_state_functions/LSF_2D_ChangingTopological.py`, `Limit_state_functions/LSF_2D_LinearLog.py` — examples A–C.
- `MCMC_samplers/MCMCBase.py`, `MCMC_samplers/CW_aMH.py`, `MCMC_samplers/aCS.py`, `MCMC_samplers/HMC.py`, `MCMC_samplers/ES.py` — locally included sampling interface and kernels.
- `Support_Fun/ReSuSProduct.py` — native probability-product postprocessing.
- `Operations_on_Logarithmic_values/logGauss.py` — the existing Gaussian log-density calculation.

## Quick start
Run from this repository's root:

```bash
python -m pip install -r requirements.txt
python drive_resus.py --example A --n 1000 --seed 411 --sampler ES --profile paper
python drive_resus.py --example C --n 1000 --seed 411 --sampler aCS --profile paper
```
The returned result includes `pf`, `log_pf`, `conditional_probabilities`, actual `n_calls`, `converged`, sampled levels. `converged = false` is an unfinished run, not a validated estimate. `estimator` identifies this revision as `product`.

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

## Method notes
- **Implementation:** each transition contributes its empirical hybrid-event probability on the **current sampled level**, divided by the actual number of samples (which can differ from requested N). The last factor is the true-failure proportion `g <= 0`; no extra unsampled hybrid factor is included. Factors are accumulated in log space. A zero estimate is retained as `pf=0`, `log_pf=-Inf`.
- **Sampler policy:** `CW_aMH` uses seed-spread-scaled **uniform random-walk increments in whitened Gaussian-reference coordinates**. `ES` is the coordinatewise initial-angle `[0, pi]` version, not a claim of identical historical scalar-angle ESS. `aCS` starts with `sigma0=1`, `lambda0=0.6` and carries the final lambda between levels (five adaptation groups, target acceptance 0.44). `HMC` uses analytic Gaussian-reference rotation; its adaptive trajectory time is carried between levels. It is not generic physical-coordinate gradient HMC.
- **Shared interface:** `SamGen(seed_u, seed_y, Nsteps)` returns `(u, y, Ncal, BayStc)` (MATLAB uses multiple outputs); arrays are dimension/function × chain × step. `G` carries thresholds and adaptation. Local copies of common kernels keep each repository independently runnable; no sibling repository is required.
- **Outputs:** `conditional_probabilities` contains the factors, `log_normalizers` their cumulative level normalizers, and `sample_counts` actual level sizes. `n_calls` counts all LSF evaluations, including rejected proposals. Standard deviation across independent runs is not the standard error of their mean. Random seeds do not identify equal streams across languages.

## Reference & license
Binbin Li, Weili Xia and Zihan Liao. **Relaxed subset simulation for reliability estimation.** *Reliability Engineering & System Safety* 264 (2025), 111302. [DOI](https://doi.org/10.1016/j.ress.2025.111302).

Licensed under the [MIT License](LICENSE). Original copyright and source notices are retained. [Other language](https://github.com/ZihanLiao-ZJU/Relaxed-Subset-Simulation).

## Directory layout

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
