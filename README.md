# Relaxed Subset Simulation

A compact reliability-estimation toolbox using relaxed hybrid intermediate domains and multiple importance sampling (MIS).

## Reference and implementation choice

Binbin Li, Weili Xia, Zihan Liao. **Relaxed subset simulation for reliability estimation**. *Reliability Engineering & System Safety* **264** (2025), 111302. [DOI](https://doi.org/10.1016/j.ress.2025.111302).

**This implementation uses balance-heuristic MIS over samples from all sampled levels, rather than the sequential probability-product estimator in the paper.** Intermediate normalizers are also estimated with historical MIS. It organizes the supplied research-code variant; it is not a literal implementation of every setting in the paper.

The intermediate domain is the union of an LSF sublevel set and a low-Gaussian-density set. Thresholds use adjacent-order-statistic midpoints. The supplied coordinatewise angular sampler is retained, with a maximum of 20 shrink attempts; this is not the scalar-angle ESS formulation. Accepted chains advance even if another chain reaches that cap. Seeds use strict inequalities; intermediate domain membership and final failure use inclusive inequalities.

## Structure

- `SeMCS`: iteration, storage, MIS normalizers and final failure estimate.
- `ReSuS`: intermediate distributions, threshold updates, seed selection and convergence.
- `ES`: sampling within the intermediate domain.
- `drive_resus`: experiment settings and execution.
- `LSF_2D_PiecewiseLinear`, `LSF_2D_ChangingTopological`: examples A and B.

Each repository is independent; the six algorithm/example source files correspond one-to-one between languages. Examples A and B use independent standard-normal inputs. Other examples and nonnormal/Nataf transformations are not included.

## Python usage

Requires Python 3.10 or later and NumPy. From this repository directory:

```bash
python -m pip install -r requirements.txt
python drive_resus.py --example A --n 5000 --seed 1
python drive_resus.py --example B --n 5000 --seed 1
```

```python
from drive_resus import drive_resus
result = drive_resus('A', 5000, 1)
```

The result includes `pf`, `log_pf`, `converged`, actual `n_calls`, level samples, thresholds, normalizers and sample counts. `n_calls` counts every LSF evaluation, including rejected proposals. Actual samples per level may differ slightly from the requested count due to chain-length rounding. A result with `converged = false` is not a completed run; increase the level budget or inspect the model. Random seeds do not produce identical streams across MATLAB and Python. Individual estimates fluctuate; use independent repetitions for numerical studies.

Custom models must accept a dimension-by-sample matrix and return one finite LSF value per sample. Failure is `g <= 0`; inputs to this minimal implementation are independent standard normals. Set the model and dimension through `SeMCS`.

## License

MIT; the original source copyright notice is retained in `LICENSE`.
