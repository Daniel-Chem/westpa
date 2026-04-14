import numpy as np

from .base import Resampler
from .ops import resample_equal_weight


class ResidualResampler(Resampler):
    """Implements residual resampling (described `here <https://arxiv.org/abs/1806.00860>`_ in Algorithm 8.1).

    Parameters
    ----------
    seed, smallest_allowed_weight, largest_allowed_weight
        See :class:`Resampler` class documentation for details.

    """

    def resample(self, bin, target_count):
        weights = bin.weights()
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

        ideal_weight = total_weight / target_count
        return resample_equal_weight(bin, counts, new_weight=ideal_weight)
