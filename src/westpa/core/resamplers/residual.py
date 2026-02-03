import numpy as np

from .base import Resampler


class ResidualResampler(Resampler):
    """Implements the residual resampling method described by Aristoff and Zuckerman. [1]_

    Parameters
    ----------
    bin_mapper : BinMapper
        Bin mapper for assigning trajectories to bins.
    bin_target_counts : int or sequence of int
        Target number of trajectories for each bin. If an integer is provided,
        the value will be applied to all the bins. If a sequence is provided,
        its length must match ``bin_mapper.nbins``.
    seed : int or sequence of int, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.

    References
    ----------
    .. [1] D. Aristoff, D.M. Zuckerman, \
    Multiscale Model. Simul., Volume 18, Issue 2, 2020, Pages 646-673, https://doi.org/10.1137/18M1212100.

    """

    def __init__(self, *, bin_mapper, bin_target_counts, seed=None):
        super().__init__(
            bin_mapper=bin_mapper,
            bin_target_counts=bin_target_counts,
            adjust_counts=False,
            seed=seed,
        )

    def resample(self, bin, target_count):
        weights = np.array([segment.weight for segment in bin])
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

        new_segments = set()
        new_weight = total_weight / target_count
        wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))
        for segment, count in zip(bin, counts):
            for _ in range(count):
                new_segment = segment.copy(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
                new_segments.add(new_segment)

        bin.clear()
        bin.update(new_segments)

        return bin
