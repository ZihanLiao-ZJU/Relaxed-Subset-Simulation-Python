"""Sequential Monte Carlo orchestration and Re-SuS probability products (Eq. 24)."""
import numpy as np
from ReSuS import ReSuS
from Support_Fun.ReSuSProduct import ReSuSProduct
from Operations_on_Logarithmic_values.logGauss import logGauss
from MCMC_samplers.ES import ES
from MCMC_samplers.aCS import aCS
from MCMC_samplers.HMC import HMC
from MCMC_samplers.CW_aMH import CW_aMH


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
        samples, values, distributions, log_normalizers, counts = [], [], [], [], []
        conditional_probabilities = []
        converged = False
        for _ in range(self.max_levels):
            lp = logGauss(x, self.dimension)
            samples.append(x); values.append(g)
            distributions.append(dist); log_normalizers.append(log_z); counts.append(len(g))
            next_dist = dist.update(g, lp, self.p0)
            if next_dist.converged():
                converged = True
                break
            if len(samples) == self.max_levels:
                break
            # Eq. (23): conditional hybrid probability on this level only.
            probability = float(np.mean(next_dist.contains(g, lp)))
            if probability <= 0:
                raise RuntimeError('Intermediate distribution has no supported samples')
            conditional_probabilities.append(probability)
            next_log_z = log_z + np.log(probability)
            seed_x, seed_g = next_dist.select_seeds(x, g, lp, self.n_samples, rng)
            length = int(np.floor(self.n_samples/seed_x.shape[1]+0.5))
            lp_seed = logGauss(seed_x, self.dimension)
            packed = np.vstack((np.where(next_dist.contains(seed_g, lp_seed), 0., -np.inf), lp_seed, lp_seed, seed_g))
            self.sampler.BayStc = next_dist
            x, packed, used, next_dist = self.sampler.SamGen(seed_x, packed, length)
            x = x.reshape(self.dimension, -1, order="F")
            g = packed[3].reshape(-1, order="F")
            calls += used
            dist, log_z = next_dist, next_log_z
        # Eq. (21)/(24): replace the un-sampled next hybrid event with true failure.
        pf, log_pf, final_probability = ReSuSProduct(g, log_z)
        conditional_probabilities.append(final_probability)
        return dict(estimator="product", conditional_probabilities=np.array(conditional_probabilities), profile=self.profile, sampler=type(self.sampler).__name__, pf=pf, log_pf=log_pf, converged=converged, n_calls=calls, samples=samples,
                    values=values, thresholds=np.array([[d.g_threshold, d.logpdf_threshold] for d in distributions]),
                    log_normalizers=np.array(log_normalizers), sample_counts=np.array(counts))
