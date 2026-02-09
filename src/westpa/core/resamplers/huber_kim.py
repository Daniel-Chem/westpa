import logging
import math

import numpy as np

from .base import Resampler

logger = logging.getLogger(__name__)


class HuberKimResampler(Resampler):
    """Implements the weighted ensemble method of Huber and Kim. [1]_

    Parameters
    ----------
    smallest_allowed_weight, largest_allowed_weight, seed
        See :class:`Resampler` class documentation for details.
    adjust_counts : bool, default True
        Whether to adjust the number of segments in occupied bins to exactly
        match the target count. This is a modification of the original
        Huber-Kim method, which only ensures that the number of segments is
        close to the target count.
        The adjustment is made either by iteratively merging
        the two lowest-weight segments (if above the target count) or by
        iteratively splitting the highest-weight segment (if below the target
        count).

    References
    ----------
    .. [1] G.A. Huber, S. Kim,
       Biophysical Journal, Volume 70, Issue 1, 1996, Pages 97-110, ISSN 0006-3495,
       https://doi.org/10.1016/S0006-3495(96)79552-8.

    """

    split_threshold = 2.0
    merge_cutoff = 1.0

    def __init__(
        self,
        *,
        smallest_allowed_weight=1e-310,
        largest_allowed_weight=1.0,
        seed=None,
        adjust_counts=True,
    ):
        super().__init__(
            smallest_allowed_weight=smallest_allowed_weight,
            largest_allowed_weight=largest_allowed_weight,
            seed=seed,
        )
        self.adjust_counts = bool(adjust_counts)

    def _split_by_weight(self, bin, ideal_weight):
        # Split walkers with weight > split_threshold * ideal_weight.
        index = bin.bisect_weights(self.split_threshold * ideal_weight)
        to_split = bin.segments[index:]
        for segment in to_split:
            self.split(segment, bin, m=math.ceil(segment.weight / ideal_weight))

    def _merge_by_weight(self, bin, ideal_weight):
        # Merge sets of walkers with combined weight <= merge_cutoff * ideal_weight.
        while True:
            cumul_weight = np.cumsum(bin.weights)
            index = np.searchsorted(cumul_weight, self.merge_cutoff * ideal_weight, side='right')
            to_merge = bin.segments[:index]
            if len(to_merge) < 2:
                break
            self.merge(to_merge, bin, total_weight=cumul_weight[index - 1])

    def _adjust_count(self, bin, target_count):
        while len(bin) < target_count:
            logger.debug('adjusting counts by splitting')
            self.split(bin.segments[-1], bin, m=2)
        while len(bin) > target_count:
            logger.debug('adjusting counts by merging')
            self.merge(bin.segments[:2], bin)

    def resample(self, bin, target_count):
        ideal_weight = bin.weight / target_count
        self._split_by_weight(bin, ideal_weight)
        self._merge_by_weight(bin, ideal_weight)
        if self.adjust_counts:
            self._adjust_count(bin, target_count)
        return bin
