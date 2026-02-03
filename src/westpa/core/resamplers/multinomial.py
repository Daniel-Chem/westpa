import numpy as np

from .base import Resampler


class MultinomialResampler(Resampler):
    """Randomly selects trajectories with replacement according to their relative weight.

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

    Notes
    -----
    This resampler is equivalent to the equal weight resampler implemented by
    Plotnikov and Ahn [1]_.

    References
    ----------
    .. [1] D. Plotnikov, S.-H. Ahn, \
    J. Chem. Phys., Volume 161, 2024, Page 046101, https://doi.org/10.1063/5.0197141.

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

        counts = self.rng.multinomial(n=target_count, pvals=weights / total_weight)

        new_segments = set()
        new_weight = total_weight / target_count
        wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))
        for segment, count in zip(bin, counts):
            for _ in range(count):
                new_segment = segment.copy(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
                new_segment.add(new_segments)

        bin.clear()
        bin.update(new_segments)

        return bin
