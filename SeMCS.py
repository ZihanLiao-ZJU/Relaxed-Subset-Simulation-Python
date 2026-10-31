"""Sequential Monte Carlo orchestration and balance-heuristic MIS estimates."""
import numpy as np
from ReSuS import ReSuS
from ES import ES
from aCS import aCS
from HMC import HMC
from CW_aMH import CW_aMH


class SeMCS:
    def __init__(self, lsf, dimension, n_samples=5000, p0=0.1, max_levels=100, sampler="ES", profile="legacy"):
        n = n_samples
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('n must be an integer >= 2')
        if not 1 <= int(np.floor(n*p0+0.5)) < n:
            raise ValueError('p0 gives an invalid threshold rank')
        if dimension < 1 or int(dimension) != dimension or max_levels < 1 or int(max_levels) != max_levels:
            raise ValueError('dimension and max_levels must be positive integers')
        self.lsf, self.dimension, self.n_samples = lsf, int(dimension), n
        self.p0, self.max_levels = p0, int(max_levels)
        self.profile = profile
        factories = {"ES": ES, "aCS": aCS, "HMC": HMC, "CW_aMH": CW_aMH}
        if isinstance(sampler, str):
            if sampler not in factories:
                raise ValueError("sampler must be CW_aMH, ES, aCS, or HMC")
            self.sampler = factories[sampler]()
        else:
            self.sampler = sampler

    def run(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        if hasattr(self.sampler, "reset"):
            self.sampler.reset()
        x = rng.standard_normal((self.dimension, self.n_samples))
        g = np.asarray(self.lsf(x), dtype=float).reshape(-1)
        if len(g) != self.n_samples or not np.all(np.isfinite(g)):
            raise ValueError('LSF must return one finite value per sample')
        calls, dist, log_z = self.n_samples, ReSuS(profile=self.profile), 0.0
        dist.lsf = self.lsf
        dist.Ndim = self.dimension
        dist.G = {"L": np.eye(self.dimension)}
        self.sampler.rng = rng
        sampler_history = []
        samples, values, distributions, log_normalizers, counts = [], [], [], [], []
        converged = False
        for _ in range(self.max_levels):
            lp = -0.5*np.sum(x*x, axis=0)-self.dimension/2*np.log(2*np.pi)
            samples.append(x); values.append(g)
            distributions.append(dist); log_normalizers.append(log_z); counts.append(len(g))
            pooled_g = np.concatenate(values)
            pooled_x = np.concatenate(samples, axis=1)
            pooled_lp = -0.5*np.sum(pooled_x**2, axis=0)-self.dimension/2*np.log(2*np.pi)
            log_denom = np.full(len(pooled_g), -np.inf)
            for previous, z, count in zip(distributions, log_normalizers, counts):
                term = np.where(previous.contains(pooled_g, pooled_lp), np.log(count)-z, -np.inf)
                log_denom = np.logaddexp(log_denom, term)
            next_dist = dist.update(g, lp, self.p0)
            if next_dist.converged():
                converged = True
                break
            if len(samples) == self.max_levels:
                break
            mask = next_dist.contains(pooled_g, pooled_lp)
            if not np.any(mask):
                raise RuntimeError('Intermediate distribution has no supported samples')
            next_log_z = np.logaddexp.reduce(-log_denom[mask])
            seed_x, seed_g = next_dist.select_seeds(x, g, lp, self.n_samples, rng)
            length = int(np.floor(self.n_samples/seed_x.shape[1]+0.5))
            lp_seed = -.5*np.sum(seed_x*seed_x, axis=0)-self.dimension/2*np.log(2*np.pi)
            packed = np.vstack((np.where(next_dist.contains(seed_g, lp_seed), 0., -np.inf), lp_seed, lp_seed, seed_g))
            self.sampler.BayStc = next_dist
            previous_state = dict(next_dist.G)
            x, packed, used, next_dist = self.sampler.SamGen(seed_x, packed, length)
            sampler_history.append({"initial": previous_state, "final": dict(next_dist.G)})
            x = x.reshape(self.dimension, -1, order="F")
            g = packed[3].reshape(-1, order="F")
            calls += used
            dist, log_z = next_dist, next_log_z
        failures = pooled_g <= 0
        log_pf = float(np.logaddexp.reduce(-log_denom[failures])) if np.any(failures) else -np.inf
        pf = float(np.exp(log_pf))
        return dict(sampler_history=sampler_history, profile=self.profile, sampler=type(self.sampler).__name__, pf=pf, log_pf=log_pf, converged=converged, n_calls=calls, samples=samples,
                    values=values, thresholds=np.array([[d.g_threshold, d.logpdf_threshold] for d in distributions]),
                    log_normalizers=np.array(log_normalizers), sample_counts=np.array(counts))
