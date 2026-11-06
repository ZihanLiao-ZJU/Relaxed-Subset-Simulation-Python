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
        self.Ndim = None
        self.lsf = None
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
            obj.g_threshold, obj.logpdf_threshold = sg[k - 1], sp[k - 1]
        else:
            obj.g_threshold, obj.logpdf_threshold = (sg[k - 1] + sg[k]) / 2, (sp[k - 1] + sp[k]) / 2
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
        lengths = np.floor(n / np.arange(1, n + 1) + 0.5)
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

    def start_run(self, owner, sampler):
        n = owner.NumSam
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('n must be an integer >= 2')
        if not 1 <= int(np.floor(n * owner.ConPrb + 0.5)) < n:
            raise ValueError('p0 gives an invalid threshold rank')
        if (
            self.Ndim < 1
            or int(self.Ndim) != self.Ndim
            or owner.MaxIte < 1
            or int(owner.MaxIte) != owner.MaxIte
        ):
            raise ValueError('dimension and max_levels must be positive integers')
        if owner.G:
            raise ValueError('ReSuS selects its hybrid thresholds adaptively')
        sampler = copy.copy(sampler)
        if hasattr(sampler, "reset"):
            sampler.reset()
        rng = np.random.default_rng() if sampler.rng is np.random else sampler.rng
        sampler.rng = rng
        model = copy.copy(self)
        model.G = {
            "L": np.eye(self.Ndim),
            "g_threshold": np.inf,
            "logpdf_threshold": np.inf,
            "log_normalizer": 0.0,
        }
        return model, sampler

    def sample_level(self, sampler, n, xseed, yseed, steps, level):
        if level == 0:
            x = sampler.rng.standard_normal((self.Ndim, n))
            g = np.asarray(self.lsf(x), dtype=float).reshape(-1)
            if len(g) != n or not np.all(np.isfinite(g)):
                raise ValueError('LSF must return one finite value per sample')
            calls = n
            model = self
            lp = logGauss(x, self.Ndim)
            y = np.vstack((np.where(model.contains(g, lp), 0.0, -np.inf), lp, lp, g))
            shape = (len(g), 1)
        else:
            sampler.BayStc = self
            x, y, calls, model = sampler.SamGen(xseed, yseed, steps)
            shape = (xseed.shape[1], steps)
            x = x.reshape(self.Ndim, -1, order="F")
            y = y.reshape(4, -1, order="F")
        return x, y, calls, model, shape

    def select_run_seeds(self, x, y, n, rng, previous):
        model = self
        probability = float(np.mean(model.contains(y[3], y[2])))
        if probability <= 0:
            raise RuntimeError('Intermediate distribution has no supported samples')
        model.G = dict(
            model.G, log_normalizer=previous.G["log_normalizer"] + np.log(probability)
        )
        seed_x, seed_g = model.select_seeds(x, y[3], y[2], n, rng)
        length = int(np.floor(n / seed_x.shape[1] + 0.5))
        lp = logGauss(seed_x, self.Ndim)
        seed_y = np.vstack((np.where(model.contains(seed_g, lp), 0.0, -np.inf), lp, lp, seed_g))
        return seed_x, seed_y, length, model, probability

    @property
    def g_threshold(self):
        return self.G["g_threshold"]
    @g_threshold.setter
    def g_threshold(self, value):
        self.G = dict(self.G, g_threshold=value)
    @property
    def logpdf_threshold(self):
        return self.G["logpdf_threshold"]
    @logpdf_threshold.setter
    def logpdf_threshold(self, value):
        self.G = dict(self.G, logpdf_threshold=value)

    RunOptions = dict(
        Defaults=dict(NumSam=5000, ConPrb=0.1, MaxIte=100, Verbose=False),
        CountPolicy="scalar", LoopOffset=0, PostCapOffset=0,
        StopPhase="before_seeds", PresetMode="adaptive", UpdateOutputs=1,
        SeedFunction=select_run_seeds, SeedInputs="transition", StateHistory="sample",
        OutputFields=("conditional_probabilities", "profile", "sampler"),
    )

