"""Hybrid LSF / Gaussian-tail intermediate distributions."""
import numpy as np


class ReSuS:
    def __init__(self, g_threshold=np.inf, logpdf_threshold=np.inf):
        self.g_threshold = float(g_threshold)
        self.logpdf_threshold = float(logpdf_threshold)

    def contains(self, g, logpdf):
        return (g <= self.g_threshold) | (logpdf <= self.logpdf_threshold)

    def update(self, g, logpdf, p0):
        k = int(np.floor(len(g) * p0 + 0.5))
        if not 1 <= k < len(g):
            raise ValueError('p0 must select at least one but fewer than all samples')
        sg, sp = np.sort(g), np.sort(logpdf)
        return ReSuS((sg[k-1]+sg[k])/2, (sp[k-1]+sp[k])/2)

    def select_seeds(self, x, g, logpdf, n, rng):
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
