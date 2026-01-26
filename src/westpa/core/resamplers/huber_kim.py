import inspect
import logging
import math
import operator

import numpy as np

from .base import Resampler

logger = logging.getLogger(__name__)


class HuberKimResampler(Resampler):
    """Implements a modified version of the Huber-Kim algorithm. [1]_

    Parameters
    ----------
    bin_mapper : BinMapper
        Bin mapper for assigning trajectory segments to bins.
    bin_target_counts : int or sequence of int
        Number of walkers to allocate to each bin. If an integer is provided,
        the value will be applied to all the bins. If a sequence is provided,
        its length must match ``bin_mapper.nbins``.
    split_threshold : float, default 2.0
        Threshold for splitting, in units of the "ideal weight" (the total
        weight of a bin divided by the target count).
    merge_cutoff : float, default 1.0
        Cutoff for merging, in units of the "ideal weight".
    adjust_counts : bool, default True
        If True, the number of walkers in a bin will be adjusted to exactly
        match the target count, except when this would violate the `min_weight`
        or `max_weight` constraints.
    min_weight : float, default 1e-100
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.
    seed : int, sequence of int, or numpy.random.Generator, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.

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
        split_threshold=2.0,
        merge_cutoff=1.0,
        adjust_counts=True,
        min_weight=1e-100,
        max_weight=1.0,
        seed=None,
    ):
        self.bin_mapper = None
        self.bin_target_counts = None
        self._update_bins(bin_mapper, bin_target_counts)

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.adjust_counts = adjust_counts
        self.split_threshold = split_threshold
        self.merge_cutoff = merge_cutoff

        super().__init__(seed=seed)

    def _update_bins(self, bin_mapper, bin_target_counts):
        if isinstance(bin_target_counts, int):
            bin_target_counts = np.repeat(bin_target_counts, bin_mapper.nbins)
        else:
            bin_target_counts = np.asarray(bin_target_counts, dtype=int)
            if not len(bin_target_counts) == bin_mapper.nbins:
                raise ValueError("length of 'bin_target_counts' must equal the number of bins")

        if (bin_target_counts < 1).any():
            raise ValueError("'bin_target_counts' must be positive")

        self.bin_mapper = bin_mapper
        self.bin_target_counts = bin_target_counts

    def resample(self, segments):
        bins_by_index = self.bin_mapper.map(segments)

        weight_getter = operator.attrgetter('weight')

        # Adapted from WEDriver._run_we():
        for index, bin_ in bins_by_index.items():
            target_count = self.bin_target_counts[index]

            segments = np.array(sorted(bin_, key=weight_getter))
            weights = np.array(list(map(weight_getter, segments)))

            ideal_weight = weights.sum() / target_count

            # Split walkers with weight > split_threshold * ideal_weight.
            to_split = weights > self.split_threshold * ideal_weight
            for segment in segments[to_split]:
                bin_.remove(segment)
                m = int(math.ceil(segment.weight / ideal_weight))
                new_segments = self.split(segment, m)
                bin_.update(new_segments)
            segments = segments[~to_split]
            weights = weights[~to_split]

            # Merge sets of walkers with cumulative weight <= merge_cutoff * ideal_weight.
            cumulative_weight = np.add.accumulate(weights)
            to_merge = cumulative_weight <= self.merge_cutoff * ideal_weight
            while sum(to_merge) >= 2:
                bin_.difference_update(segments[to_merge])
                new_segment = self.merge(segments[to_merge])
                bin_.add(new_segment)
                segments = segments[~to_merge]
                cumulative_weight = cumulative_weight[~to_merge] - new_segment.weight
                to_merge = cumulative_weight <= self.merge_cutoff * ideal_weight

            # Adjust counts. TODO: Refactor to avoid repeated sorts.
            if self.adjust_counts:
                while len(bin_) < target_count:
                    logger.debug('adjusting counts by splitting')
                    segments = sorted(bin_, key=weight_getter)
                    bin_.remove(segments[-1])
                    new_segments = self.split(segments[-1])  # split largest walker in 2
                    bin_.update(new_segments)
                while len(bin_) > target_count:
                    logger.debug('adjusting counts by merging')
                    segments = sorted(bin_, key=weight_getter)
                    bin_.difference_update(segments[:2])  # merge 2 smallest walkers
                    new_segment = self.merge(segments[:2])
                    bin_.add(new_segment)

            # Apply weight thresholds.
            segments = np.array(sorted(bin_, key=weight_getter))
            weights = np.array(list(map(weight_getter, segments)))
            to_split = weights > self.max_weight
            for segment in segments[to_split]:
                bin_.remove(segment)
                m = int(math.ceil(segment.weight / self.max_weight))
                new_segments = self.split(segment, m)
                bin_.update(new_segments)
            to_merge = weights < self.min_weight
            while sum(to_merge) >= 2:
                bin_.difference_update(segments[to_merge])
                new_segment = self.merge(segments[to_merge])
                bin_.add(new_segment)
                segments = np.array(sorted(bin_, key=weight_getter))
                weights = np.array(list(map(weight_getter, segments)))
                to_merge = weights < self.min_weight

            for segment in bin_:
                if not (self.min_weight <= segment.weight <= self.max_weight):
                    logger.warning(f'Unable to fulfill weight constraints for {segment}.')

        return set.union(*bins_by_index.values())

    def __repr__(self):
        args = []
        for name in inspect.signature(self.__init__).parameters:
            value = getattr(self, name)
            if name == 'bin_target_counts':
                if np.unique(value).size == 1:  # if counts are all the same
                    value = value[0].item()  # reduce to an integer
                else:
                    value = value.tolist()
            args.append(f'{name}={value!r}')
        return type(self).__name__ + '(' + ', '.join(args) + ')'
