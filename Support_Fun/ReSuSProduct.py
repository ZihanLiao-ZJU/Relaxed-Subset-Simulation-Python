"""Native ReSuS probability-product postprocessing."""

import numpy as np


def ReSuSProduct(run):
    states = run["g"][:run["Nite"]]
    values = [y.reshape(y.shape[0], -1, order="F")[3] for y in run["y"]]
    factors, log_normalizers = [], [0.0]
    for j, state in enumerate(states[1:]):
        previous = run["y"][j].reshape(run["y"][j].shape[0], -1, order="F")
        probability = float(np.mean(
            (previous[3] <= state["g_threshold"]) | (previous[2] <= state["logpdf_threshold"])
        ))
        factors.append(probability)
        log_normalizers.append(log_normalizers[-1] + np.log(probability))
    g = values[-1]
    log_z = log_normalizers[-1]
    final_probability = float(np.mean(g <= 0))
    log_pf = float(log_z + np.log(final_probability)) if final_probability > 0 else -np.inf
    pf = float(np.exp(log_pf))
    factors.append(final_probability)
    return dict(
        estimator="product",
        conditional_probabilities=np.asarray(factors),
        profile=run["intBay"].profile,
        sampler=run["sampler"],
        pf=pf,
        log_pf=log_pf,
        converged=run["flg_cvg"],
        n_calls=int(np.sum(run["Ncal"])),
        samples=[x.reshape(x.shape[0], -1, order="F") for x in run["x"]],
        values=values,
        thresholds=np.asarray([[g["g_threshold"], g["logpdf_threshold"]] for g in states]),
        log_normalizers=np.asarray(log_normalizers),
        sample_counts=run["Nsam"],
    )
