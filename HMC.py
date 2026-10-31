"""Analytic Gaussian HMC with rejection and paper section 2.2.2 adaptation.

Initial time pi/4 and five groups are explicit implementation settings;
the paper does not specify those two numerical settings.
"""
import numpy as np


class HMC:
    def __init__(self, initial_time=np.pi / 4, groups=5, carry_state=False):
        self.initial_time, self.groups = float(initial_time), int(groups)
        self.carry_state = bool(carry_state)
        self.reset()

    def reset(self):
        self.time = self.initial_time

    def sample(self, lsf, distribution, xseed, gseed, chain_length, rng):
        d, nc = xseed.shape
        current, values = xseed.copy(), gseed.copy()
        xs = np.zeros((d, nc, chain_length)); gs = np.zeros((nc, chain_length))
        normals = rng.standard_normal((d, nc, chain_length))
        t, calls = (self.time if self.carry_state else self.initial_time), 0
        group_size = int(np.ceil(nc / self.groups))
        for start in range(0, nc, group_size):
            ids = np.arange(start, min(start + group_size, nc))
            accepted_count = 0
            for k in range(chain_length):
                candidate = np.cos(t) * current[:, ids] + np.sin(t) * normals[:, ids, k]
                g = np.asarray(lsf(candidate), dtype=float).reshape(-1)
                if len(g) != len(ids) or not np.all(np.isfinite(g)):
                    raise ValueError('LSF must return one finite value per sample')
                lp = -0.5 * np.sum(candidate**2, axis=0) - d / 2 * np.log(2 * np.pi)
                accepted = distribution.contains(g, lp)
                accepted_count += np.count_nonzero(accepted)
                current[:, ids[accepted]] = candidate[:, accepted]
                values[ids[accepted]] = g[accepted]
                xs[:, ids, k] = current[:, ids]; gs[ids, k] = values[ids]
                calls += len(ids)
            rate = accepted_count / (len(ids) * chain_length)
            if rate < 0.3:
                t = np.arcsin(np.clip(np.sin(t) * np.exp((rate - 0.3) / 2), -1, 1))
            elif rate > 0.5:
                t = np.arcsin(np.clip(np.sin(t) * np.exp((rate - 0.5) / 2), -1, 1))
        if self.carry_state:
            self.time = t
        return xs.reshape(d, -1, order='F'), gs.reshape(-1, order='F'), calls
