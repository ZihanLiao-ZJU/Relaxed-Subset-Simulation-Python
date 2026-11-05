"""Native ReSuS probability-product postprocessing."""

import numpy as np


def ReSuSProduct(run):
    factors = list(run["conditional_probabilities"])
    log_normalizers = [g["log_normalizer"] for g in run["g"]]
    g = run["y"][-1][3]
    log_z = log_normalizers[-1]
    final_probability = float(np.mean(g <= 0))
    log_pf = float(log_z + np.log(final_probability)) if final_probability > 0 else -np.inf
    pf = float(np.exp(log_pf))
    factors.append(final_probability)
    return dict(
        estimator="product",
        conditional_probabilities=np.asarray(factors),
        profile=run["profile"],
        sampler=run["sampler"],
        pf=pf,
        log_pf=log_pf,
        converged=run["flg_cvg"],
        n_calls=int(np.sum(run["Ncal"])),
        samples=run["x"],
        values=[y[3] for y in run["y"]],
        thresholds=np.asarray([[g["g_threshold"], g["logpdf_threshold"]] for g in run["g"]]),
        log_normalizers=np.asarray(log_normalizers),
        sample_counts=run["Nsam"],
    )
