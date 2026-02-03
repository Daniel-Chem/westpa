import logging
import math
import numpy as np

from .base import Resampler

logger = logging.getLogger(__name__)


class HuberKimResampler(Resampler):
    """Implements the weighted ensemble method of Huber and Kim. [1]_

    Parameters
    ----------
    bin_mapper : BinMapper
        Bin mapper for assigning trajectories to bins.
    bin_target_counts : int or sequence of int
        Target number of trajectories for each bin. If an integer is provided,
        the value will be applied to all the bins. If a sequence is provided,
        its length must match ``bin_mapper.nbins``.
    adjust_counts : bool, default True
        Whether to adjust the number of segments in occupied bins to exactly
        match the target count. This is done by iteratively merging the two
        lowest-weight segments (if above the target count) or by iteratively
        splitting the highest-weight segment (if below the target count).
    smallest_allowed_weight : float, default 1e-100
        Smallest allowed weight.
    larget_allowed_weight : float, default 1.0
        Largest allowed weight.
    seed : int or sequence of int, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.
    split_threshold : float, default 2.0
        Threshold for splitting, in units of the "ideal weight" (the total
        weight of a bin divided by the target count).
    merge_cutoff : float, default 1.0
        Cutoff for merging, in units of the "ideal weight".

    References
    ----------
    .. [1] G.A. Huber, S. Kim, \
    Biophysical Journal, Volume 70, Issue 1, 1996, Pages 97-110, ISSN 0006-3495, https://doi.org/10.1016/S0006-3495(96)79552-8.

    """

    def __init__(
        self,
        *,
        bin_mapper,
        bin_target_counts,
        adjust_counts=True,
        smallest_allowed_weight=1e-100,
        largest_allowed_weight=1.0,
        seed=None,
        split_threshold=2.0,
        merge_cutoff=1.0,
    ):
        super().__init__(
            bin_mapper=bin_mapper,
            bin_target_counts=bin_target_counts,
            adjust_counts=adjust_counts,
            smallest_allowed_weight=smallest_allowed_weight,
            largest_allowed_weight=largest_allowed_weight,
            seed=seed,
        )
        self.split_threshold = split_threshold
        self.merge_cutoff = merge_cutoff

    def resample(self, bin, target_count):
        ideal_weight = bin.weight / target_count
        self._split_by_weight(bin, ideal_weight)
        self._merge_by_weight(bin, ideal_weight)
        return bin

    def _split_by_weight(self, bin, ideal_weight):
        # Split walkers with weight > split_threshold * ideal_weight.
        index = bin.bisect_key(self.split_threshold * ideal_weight)
        to_split = bin[index:]
        for segment in to_split:
            self.split(segment, bin, m=math.ceil(segment.weight / ideal_weight))

    def _merge_by_weight(self, bin, ideal_weight):
        # Merge sets of walkers with cumulative weight <= merge_cutoff * ideal_weight.
        while True:
            cumulative_weight = np.cumsum([segment.weight for segment in bin])
            index = np.searchsorted(cumulative_weight, self.merge_cutoff * ideal_weight)
            to_merge = bin[:index]
            if len(to_merge) < 2:
                break
            self.merge(to_merge, bin)
