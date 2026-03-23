import numpy as np

from .base import Resampler


class ResidualResampler(Resampler):
    """Implements the residual resampling method described by Aristoff and Zuckerman. [1]_

    Parameters
    ----------
    smallest_allowed_weight, largest_allowed_weight, seed
        See :class:`Resampler` class documentation for details.

    References
    ----------
    .. [1] D. Aristoff, D.M. Zuckerman,
       Multiscale Model. Simul., Volume 18, Issue 2, 2020, Pages 646-673,
       https://doi.org/10.1137/18M1212100.

    """

    def resample(self, bin, target_count):
        weights = bin.weights
        total_weight = weights.sum()

        # See Algorithm 8.1 in https://arxiv.org/abs/1806.00860.
        nd = target_count * weights / total_weight
        nd_floor = np.floor(nd)
        delta = nd - nd_floor
        trials = round(delta.sum())
        if trials == 0:  # happens when bin.count == 1
            counts = [target_count]
        else:
            r = self.rng.multinomial(n=trials, pvals=delta / trials)
            counts = map(round, nd_floor + r)

        new_weight = total_weight / target_count
        wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))
        for segment, count in zip(bin, counts):
            for _ in range(count):
                yield segment.replace(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
