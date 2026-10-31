"""Hybrid LSF / Gaussian-tail intermediate distributions."""
import numpy as np


class ReSuS:
    def __init__(self, g_threshold=np.inf, logpdf_threshold=np.inf, profile="legacy"):
        if profile not in ("legacy", "paper"):
            raise ValueError("profile must be legacy or paper")
        self.profile = profile
        self.g_threshold = float(g_threshold)
        self.logpdf_threshold = float(logpdf_threshold)

    def contains(self, g, logpdf):
        return (g <= self.g_threshold) | (logpdf <= self.logpdf_threshold)

    def update(self, g, logpdf, p0):
        k = int(np.floor(len(g) * p0 + 0.5))
        if not 1 <= k < len(g):
            raise ValueError('p0 must select at least one but fewer than all samples')
        sg, sp = np.sort(g), np.sort(logpdf)
        if self.profile == "paper":
            return ReSuS(sg[k-1], sp[k-1], self.profile)
        return ReSuS((sg[k-1]+sg[k])/2, (sp[k-1]+sp[k])/2, self.profile)

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
