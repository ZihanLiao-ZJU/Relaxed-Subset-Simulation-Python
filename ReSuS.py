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

    def run_defaults(self):
        return dict(NumSam=5000, ConPrb=0.1, MaxIte=100, Verbose=False)

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
        return dict(
            owner=owner,
            model=model,
            sampler=sampler,
            limit=int(owner.MaxIte),
            rng=rng,
            x=[],
            y=[],
            g=[],
            Nsam=[],
            Nsze=[],
            Ncal=[],
            conditional_probabilities=[],
            complete=False,
            xseed=None,
            yseed=None,
            steps=1,
        )

    def sample_level(self, context, level):
        if level == 0:
            x = context["rng"].standard_normal((self.Ndim, context["owner"].NumSam))
            g = np.asarray(self.lsf(x), dtype=float).reshape(-1)
            if len(g) != context["owner"].NumSam or not np.all(np.isfinite(g)):
                raise ValueError('LSF must return one finite value per sample')
            calls = context["owner"].NumSam
            model = context["model"]
            lp = logGauss(x, self.Ndim)
            y = np.vstack((np.where(model.contains(g, lp), 0.0, -np.inf), lp, lp, g))
            shape = (len(g), 1)
        else:
            context["sampler"].BayStc = context["model"]
            x, y, calls, model = context["sampler"].SamGen(
                context["xseed"], context["yseed"], context["steps"]
            )
            shape = (context["xseed"].shape[1], context["steps"])
            x = x.reshape(self.Ndim, -1, order="F")
            y = y.reshape(4, -1, order="F")
        context["model"] = model
        record = dict(x=x, y=y, calls=calls, count=y.shape[1], shape=shape)
        context["x"].append(x)
        context["y"].append(y)
        context["g"].append(copy.deepcopy(model.G))
        context["Nsam"].append(record["count"])
        context["Nsze"].append(shape)
        context["Ncal"].append(calls)
        return record

    def advance_level(self, context, record, level):
        context["next_model"] = context["model"].update(
            record["y"][3], record["y"][2], context["owner"].ConPrb
        )

    def run_complete(self, context, record, phase):
        if phase != "before_seeds":
            return False
        if context["next_model"].converged():
            context["complete"] = True
            return True
        return len(context["x"]) == context["limit"]

    def select_run_seeds(self, context, record, level):
        model = context["next_model"]
        probability = float(np.mean(model.contains(record["y"][3], record["y"][2])))
        if probability <= 0:
            raise RuntimeError('Intermediate distribution has no supported samples')
        context["conditional_probabilities"].append(probability)
        model.G = dict(
            model.G, log_normalizer=context["model"].G["log_normalizer"] + np.log(probability)
        )
        seed_x, seed_g = model.select_seeds(
            record["x"], record["y"][3], record["y"][2], context["owner"].NumSam, context["rng"]
        )
        length = int(np.floor(context["owner"].NumSam / seed_x.shape[1] + 0.5))
        lp = logGauss(seed_x, self.Ndim)
        context["xseed"] = seed_x
        context["yseed"] = np.vstack(
            (np.where(model.contains(seed_g, lp), 0.0, -np.inf), lp, lp, seed_g)
        )
        context["steps"] = length
        context["model"] = model

    def finish_run(self, context):
        return dict(
            x=context["x"],
            y=context["y"],
            g=context["g"],
            Nsam=np.asarray(context["Nsam"]),
            Nsze=np.asarray(context["Nsze"]),
            Ncal=np.asarray(context["Ncal"]),
            Nite=len(context["x"]),
            flg_cvg=context["complete"],
            intBay=context["model"],
            conditional_probabilities=np.asarray(context["conditional_probabilities"]),
            profile=self.profile,
            sampler=type(context["sampler"]).__name__,
        )

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
