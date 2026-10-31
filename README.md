# Relaxed Subset Simulation

A compact reliability-estimation toolbox using relaxed hybrid intermediate domains and multiple importance sampling (MIS).

## Reference and implementation choice

Binbin Li, Weili Xia, Zihan Liao. **Relaxed subset simulation for reliability estimation**. *Reliability Engineering & System Safety* **264** (2025), 111302. [DOI](https://doi.org/10.1016/j.ress.2025.111302).

**This implementation uses balance-heuristic MIS over samples from all sampled levels, rather than the sequential probability-product estimator in the paper.** Intermediate normalizers are also estimated with historical MIS. It organizes the supplied research-code variant; it is not a literal implementation of every setting in the paper.

The intermediate domain is the union of an LSF sublevel set and a low-Gaussian-density set. Two explicit threshold/seed profiles are available:

- `legacy` (default): supplied-code midpoint thresholds and strict seed eligibility.
- `paper`: order-statistic thresholds, inclusive seed eligibility and the chain-count rounding in Algorithm 1. The estimator remains MIS in both profiles.

The default run retains `N=5000` and the supplied-code `legacy` outer profile.

Choose one of four samplers:

- `CW_aMH`: coordinatewise Gaussian-reference Metropolis with per-level seed-spread proposal widths.

- `ES` (default): the public SeMIS toolbox's coordinatewise angular sampler. Each coordinate starts with the bracket `[0, pi]`; rejected proposals shrink the bracket around `acos(seed / radius)`. The retry cap is 20, seeds are not counted as new chain samples, and accepted chains advance even if another reaches the cap. This is not the separate scalar-angle ESS implementation.
- `aCS`: unit base proposal standard deviation (`sigma0 = 1`), initial `lambda = 0.6`, target acceptance 0.44 and five chain groups. The effective proposal scale is `min(lambda * sigma0, 1)`. Lambda updates within each level and the next level inherits the preceding level's final lambda. The adaptation group index starts at 1 within each level. A fresh independent simulation resets lambda to 0.6.
- `HMC`: the shared reference implementation; analytic Gaussian dynamics and rejection with the acceptance-rate adjustment described in the paper. Trajectory time is initialized to `pi/4` and adapted over five groups, with final time passed to the next level. These initialization/group settings are explicit implementation choices rather than recovered historical parameters. A fresh simulation resets time. The arcsine argument is bounded to its real domain.

The `paper` threshold/seed profile does not change MIS into the paper's probability-product estimator. All sampling implementations are included locally; no other repository is required.

## Structure

- `SeMCS`: iteration, storage, MIS normalizers and final failure estimate.
- `ReSuS`: intermediate distributions, threshold updates, seed selection and convergence.
- `CW_aMH`, `ES`, `aCS`, `HMC`: sampling within the intermediate domain.
- `drive_resus`: experiment settings and execution.
- `LSF_2D_PiecewiseLinear`, `LSF_2D_ChangingTopological`, `LSF_2D_LinearLog`: examples A, B and C.

Each repository is independent; the algorithm/example source files correspond one-to-one between languages. Examples A-C use independent standard-normal inputs. Example C follows paper Eq. (26), using `5 - x1` without the `0.03` scale in the earlier supplied source. Only A-C are included; nonnormal/Nataf transformations are not included.

## Python usage

Requires Python 3.10 or later and NumPy. From this repository directory:

```bash
python -m pip install -r requirements.txt
python drive_resus.py --example A --seed 1
python drive_resus.py --example B --seed 1
python drive_resus.py --example C --n 1000 --seed 1 --sampler ES --profile paper

# Explicit paper threshold/seed profile, with the requested MIS estimator:
python drive_resus.py --example A --n 1000 --seed 1 --sampler ES --profile paper
python drive_resus.py --example A --n 1000 --seed 1 --sampler aCS --profile paper
python drive_resus.py --example B --n 1000 --seed 1 --sampler HMC --profile paper
```

```python
from drive_resus import drive_resus
result = drive_resus('A')  # N=5000, ES, legacy profile
```

The result includes the selected `sampler` and `profile`, `pf`, `log_pf`, `converged`, actual `n_calls`, level samples, thresholds, normalizers and sample counts. `sampler_history` records each transition's initial and final distribution state (including aCS lambda). `n_calls` counts every LSF evaluation, including rejected proposals. Actual samples per level may differ slightly from the requested count due to chain-length rounding. A result with `converged = false` is not a completed run; increase the level budget or inspect the model. Random seeds do not produce identical streams across MATLAB and Python. Individual estimates fluctuate; use independent repetitions for numerical studies.

Custom models must accept a dimension-by-sample matrix and return one finite LSF value per sample. Failure is `g <= 0`; inputs to this minimal implementation are independent standard normals. Set the model and dimension through `SeMCS`.

## License

MIT; the original source copyright notice is retained in `LICENSE`.


## Unified MCMC interface

Each standalone toolbox includes `CW_aMH`, `aCS`, `HMC`, and `ES`. The original
outer algorithm and default sampler remain unchanged; choose the sampler in the
driver (Python CLI: `--sampler`). Example: `drive_resus("A", n=1000, sampler="CW_aMH")`.
All four expose `BayStc` and the same call:
`[u, y, Ncal, BayStc] = SamGen(u_seed, y_seed, Ns)` (Python returns a tuple).
`SamGen(Nchains, Nsteps)` initializes from the Gaussian reference. Arrays are
`[Ndim, Nchains, Nsteps]`; packed outputs are `[log_likelihood; log_reference; Y]`.
Dimension comes from the model, not from the two-dimensional examples.

Distribution state `G` carries model parameters and sampler adaptation. Legacy
numeric thresholds are stored in `G.parameters` (Python `G["parameters"]`);
existing named distribution parameters retain their names. aCS uses initial
`sigma0=1`, `lambda0=0.6`, five adaptation groups, and target acceptance 0.44.
The final `G.lambda` of one level initializes the next, rather than resetting.
ES uses the coordinatewise initial-angle `[0, pi]` version, with no stored seed
at the start of a conditional chain. HMC uses analytic Gaussian rotation in
Gaussian/Nataf reference coordinates, followed by the likelihood-factor MH test;
it is not a generic physical-coordinate gradient-HMC implementation. `G.L` or
`G.L_pi` is the Gaussian reference Cholesky factor, not the covariance matrix.
HMC's final trajectory time is likewise carried in `G.HMC_time`.

Custom nonnormal physical models must be mapped into Gaussian/Nataf coordinates
before these kernels are used. Arbitrary dimensional arrays are supported; the
supplied benchmark formulas retain their documented dimensions. Interface and
smoke checks do not establish statistical reproduction of every paper result.
