import abc
import logging
import math
import operator
import secrets

import numpy as np

from westpa.core.binning import Bin
from westpa.core.we_driver import ConsistencyError

logger = logging.getLogger(__name__)
weight_getter = operator.attrgetter('weight')


class Resampler(abc.ABC):
    """Base class for resamplers. Subclasses must implement the
    :meth:`resample` method, which specifies how to resample the trajectory
    segments in a given bin.

    Parameters
    ----------
    smallest_allowed_weight : float, default 1e-310
        Smallest allowed weight.
    larget_allowed_weight : float, default 1.0
        Largest allowed weight.
    seed : int or sequence of int, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.

    Attributes
    ----------
    smallest_allowed_weight : float
    largest_allowed_weight : float
    rng : numpy.random.Generator

    """

    @abc.abstractmethod
    def resample(self, bin, target_count):
        """Resample the trajectory segments in a given bin.

        Parameters
        ----------
        bin : Bin
            Bin to resample from.
        target_count : int
            Target number of segments for the bin.

        Returns
        -------
        iterable of Segment
            Set of resampled segments.

        """
        ...

    def __init__(
        self,
        *,
        smallest_allowed_weight=1e-310,
        largest_allowed_weight=1.0,
        seed=None,
    ):
        if not (0 < smallest_allowed_weight < 1):
            raise ValueError("'smallest_allowed_weight' must be between 0 and 1")
        if not (smallest_allowed_weight < largest_allowed_weight <= 1):
            raise ValueError("'largest_allowed_weight' must be between 'smallest_allowed_weight' and 1")

        seed = seed if seed is not None else secrets.randbits(128)
        logger.info(f'{seed=}')

        self._smallest_allowed_weight = smallest_allowed_weight
        self._largest_allowed_weight = largest_allowed_weight
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def split(segment, bin, m=2):
        """Split a trajectory segment into two or more copies.

        Parameters
        ----------
        segment : Segment
            Segment to split.
        bin : set of Segment
            Bin the segment belongs to.
        m : int, default 2
            Number of copies to split `segment` into.

        """
        if not isinstance(m, int):
            raise TypeError("'m' must be an integer")
        if not m >= 2:
            raise ValueError("'m' must be greater than or equal to 2")

        new_weight = segment.weight / m
        new_segments = {segment.copy(weight=new_weight) for _ in range(m)}

        bin.remove(segment)
        bin |= new_segments

    def merge(self, segments, bin, total_weight=None):
        """Merge multiple trajectory segments into a single segment. The
        surviving segment is chosen randomly according to weight.

        Parameters
        ----------
        segments : iterable of Segment
            Segments to merge.
        bin : set of Segment
            Bin the segments belong to.
        total_weight : float, optional
            Combined weight of `segments`. If not passed, the value will be
            computed by this method.

        """
        segments = list(segments)
        weights = np.array([segment.weight for segment in segments])

        if total_weight is None:
            total_weight = weights.sum()

        segment = self.rng.choice(segments, p=weights / total_weight)
        new_segment = segment.copy(
            weight=total_weight,
            wtg_parent_ids=set.union(*(segment.wtg_parent_ids for segment in segments)),
        )

        bin -= segments
        bin.add(new_segment)

    @property
    def smallest_allowed_weight(self):
        """Minimum allowed weight."""
        return self._smallest_allowed_weight

    @property
    def largest_allowed_weight(self):
        """Maximum allowed weight."""
        return self._largest_allowed_weight

    @property
    def rng(self):
        """Pseudo-random number generator."""
        return self._rng

    def _split_by_threshold(self, bin):
        index = bin.bisect_weights(self.largest_allowed_weight)
        to_split = bin.segments[index:]
        for segment in to_split:
            m = math.ceil(segment.weight / self.largest_allowed_weight)
            self.split(segment, bin, m=m)

    def _merge_by_threshold(self, bin):
        while True:
            index = bin.bisect_weights(self.smallest_allowed_weight)
            to_merge = bin.segments[:index]
            if len(to_merge) < 2:
                return
            self.merge(to_merge, bin)

    def __call__(self, bin, target_count):
        bin_weight = bin.weight

        segments = list(self.resample(bin, target_count))

        weights = np.array([segment.weight for segment in segments])
        if (weights <= 0).any():
            raise ConsistencyError('weights must be greater than 0')
        if not math.isclose(weights.sum(), bin_weight, abs_tol=1e-12):
            raise ConsistencyError('resampling must preserve the total weight of the bin')

        bin = Bin(segments, label=bin.label)

        self._split_by_threshold(bin)
        self._merge_by_threshold(bin)
        for segment in bin:
            if not (self.smallest_allowed_weight <= segment.weight <= self.largest_allowed_weight):
                logger.warning(
                    f'Unable to fulfill weight threshold conditions for {segment}. '
                    'The given threshold range is likely too small.'
                )

        return bin
