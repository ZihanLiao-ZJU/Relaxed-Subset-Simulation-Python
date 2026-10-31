"""Adaptive conditional sampling: unit base scale and inter-level lambda state."""
import numpy as np


class aCS:
    def __init__(self, scale=0.6, target_acceptance=0.44, groups=5):
        self.scale, self.target_acceptance, self.groups = float(scale), float(target_acceptance), int(groups)
        self.reset()

    def reset(self):
        """Reset adaptation only at the start of an independent simulation."""
        self.current_lambda = self.scale
        self.level_history = []

    def sample(self, lsf, distribution, xseed, gseed, chain_length, rng):
        d, nc = xseed.shape
        current, values = xseed.copy(), gseed.copy()
        xs = np.zeros((d, nc, chain_length)); gs = np.zeros((nc, chain_length))
        normals = rng.standard_normal((d, nc, chain_length))
        # Match the supplied sampler's RNG allocation (indicator MH uniforms).
        uniforms = rng.uniform(size=(nc, chain_length))
        sigma0 = np.ones(d)
        scale, calls = self.current_lambda, 0
        initial_lambda = scale
        updates = []
        group_size = int(np.ceil(nc / self.groups))
        for iteration, start in enumerate(range(0, nc, group_size), 1):
            ids = np.arange(start, min(start + group_size, nc))
            sigma = np.minimum(scale * sigma0, 1.0)
            rho = np.sqrt(1 - sigma**2)
            accepted_count = 0
            for k in range(chain_length):
                candidate = rho[:, None] * current[:, ids] + sigma[:, None] * normals[:, ids, k]
                g = np.asarray(lsf(candidate), dtype=float).reshape(-1)
                if len(g) != len(ids) or not np.all(np.isfinite(g)):
                    raise ValueError('LSF must return one finite value per sample')
                lp = -0.5 * np.sum(candidate**2, axis=0) - d / 2 * np.log(2 * np.pi)
                accepted = distribution.contains(g, lp) | (uniforms[ids, k] == 0)
                accepted_count += np.count_nonzero(accepted)
                current[:, ids[accepted]] = candidate[:, accepted]
                values[ids[accepted]] = g[accepted]
                xs[:, ids, k] = current[:, ids]; gs[ids, k] = values[ids]
                calls += len(ids)
            rate = accepted_count / (len(ids) * chain_length)
            scale = np.exp(np.log(scale) + (rate - self.target_acceptance) / np.sqrt(iteration))
            updates.append(dict(group=iteration, acceptance=float(rate), final_lambda=float(scale)))
        self.current_lambda = scale
        self.level_history.append(dict(initial_lambda=float(initial_lambda), final_lambda=float(scale), groups=updates))
        return xs.reshape(d, -1, order='F'), gs.reshape(-1, order='F'), calls
