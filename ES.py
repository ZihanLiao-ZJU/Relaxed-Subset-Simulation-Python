"""Coordinatewise angular sampler retained from the supplied ES source."""
import numpy as np


class ES:
    def __init__(self, max_tries=20):
        self.max_tries = max_tries

    def sample(self, lsf, distribution, xseed, gseed, chain_length, rng):
        d, nc = xseed.shape
        current, values = xseed.copy(), gseed.copy()
        xs, gs = [], []
        calls = 0
        normals = rng.standard_normal((d, nc, chain_length))
        for step in range(chain_length):
            radius = np.sqrt(current**2 + normals[:, :, step]**2)
            ratio = np.divide(current, radius, out=np.zeros_like(current), where=radius > 0)
            original_angle = np.arccos(np.clip(ratio, -1, 1))
            low, high = np.zeros_like(current), np.full_like(current, np.pi)
            pending = np.arange(nc)
            proposal_state, proposal_g = current.copy(), values.copy()
            for _ in range(self.max_tries):
                angle = rng.uniform(size=(d, len(pending))) * (high[:, pending]-low[:, pending]) + low[:, pending]
                candidate = np.cos(angle) * radius[:, pending]
                g = np.asarray(lsf(candidate), dtype=float).reshape(-1)
                if len(g) != len(pending) or not np.all(np.isfinite(g)):
                    raise ValueError('LSF must return one finite value per sample')
                calls += len(pending)
                lp = -0.5*np.sum(candidate**2, axis=0)-d/2*np.log(2*np.pi)
                accepted = distribution.contains(g, lp)
                proposal_state[:, pending[accepted]] = candidate[:, accepted]
                proposal_g[pending[accepted]] = g[accepted]
                rejected = ~accepted
                if not np.any(rejected):
                    break
                ids, angles = pending[rejected], angle[:, rejected]
                above = angles > original_angle[:, ids]
                low[:, ids] = np.where(above, low[:, ids], angles)
                high[:, ids] = np.where(above, angles, high[:, ids])
                pending = ids
            # Advance accepted chains even when another chain hits the attempt cap.
            current, values = proposal_state, proposal_g
            xs.append(current.copy()); gs.append(values.copy())
        return np.concatenate(xs, axis=1), np.concatenate(gs), calls
