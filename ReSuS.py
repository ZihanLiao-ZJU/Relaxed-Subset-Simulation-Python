"""Hybrid LSF / Gaussian-tail intermediate distributions."""
import numpy as np
import copy
from Operations_on_Logarithmic_values.logGauss import logGauss


class ReSuS:
    def __init__(self, g_threshold=np.inf, logpdf_threshold=np.inf, profile="legacy"):
        if profile not in ("legacy", "paper"):
            raise ValueError("profile must be legacy or paper")
        self.profile = profile
        self.G = {}
        self.Nfun = 4
        self.g_threshold = float(g_threshold)
        self.logpdf_threshold = float(logpdf_threshold)

    def contains(self, g, logpdf):
        return (g <= self.g_threshold) | (logpdf <= self.logpdf_threshold)

    def update(self, g, logpdf, p0):
        k = int(np.floor(len(g) * p0 + 0.5))
        if not 1 <= k < len(g):
            raise ValueError('p0 must select at least one but fewer than all samples')
        sg, sp = np.sort(g), np.sort(logpdf)
        obj = copy.copy(self)
        obj.G = dict(self.G)
        if self.profile == "paper":
            obj.g_threshold, obj.logpdf_threshold = sg[k-1], sp[k-1]
        else:
            obj.g_threshold, obj.logpdf_threshold = (sg[k-1]+sg[k])/2, (sp[k-1]+sp[k])/2
        return obj

    def select_seeds(self, x, g, logpdf, n, rng):
        if self.profile == "paper":
            eligible = np.flatnonzero(self.contains(g, logpdf))
            if not len(eligible):
                raise RuntimeError("No eligible seeds")
            count = int(np.floor(n / np.ceil(n / len(eligible)) + 0.5))
            chosen = rng.choice(eligible, size=count, replace=False)
            return x[:, chosen], g[chosen]
        eligible = np.flatnonzero((g < self.g_threshold) | (logpdf < self.logpdf_threshold))
        lengths = np.floor(n / np.arange(1, n+1) + 0.5)
        counts = np.floor(n / lengths + 0.5).astype(int)
        counts = counts[counts <= len(eligible)]
        if not len(counts):
            raise RuntimeError('No eligible seeds; inspect LSF ties or sample size')
        chosen = rng.choice(eligible, size=counts[-1], replace=False)
        return x[:, chosen], g[chosen]

    def converged(self):
        return self.g_threshold <= 0

    def EvlY(self, u):
        obj = copy.copy(self)
        values = np.asarray(self.lsf(u), dtype=float).reshape(1, -1)
        if values.shape[1] != u.shape[1] or not np.all(np.isfinite(values)):
            raise ValueError("LSF must return one finite value per column")
        obj.Y = np.vstack((logGauss(u, self.Ndim), values))
        obj.Ncal = u.shape[1]
        return obj

    def EvlLKF(self):
        return np.where(self.contains(self.Y[1], self.Y[0]), 0., -np.inf)

    def EvlPDF(self):
        return self.Y[0]
