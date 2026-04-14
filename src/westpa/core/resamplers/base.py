import abc
import logging
import math
import operator
import secrets

import numpy as np

from westpa.core.we_driver import ConsistencyError
from .ops import split, merge

logger = logging.getLogger(__name__)
weight_getter = operator.attrgetter('weight')


class Resampler(abc.ABC):
    """Base class for resamplers. Subclasses must implement the :meth:`resample` method.

    Parameters
    ----------
    seed : int or sequence of int, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.
    smallest_allowed_weight : float, default 1e-310
        Minimum weight threshold. Segments with weights below this value after
        calling :meth:`resample` will be merged into a single segment.
    largest_allowed_weight : float, default 1.0
        Maximum weight threshold. Segments with weights above this value after
        calling :meth:`resample` will be split into multiple copies.

    Attributes
    ----------
    rng : numpy.random.Generator
        Pseudo-random number generator.
    smallest_allowed_weight : float
        Minimum weight threshold.
    largest_allowed_weight : float
        Maximum weight threshold.

    """

    @abc.abstractmethod
    def resample(self, bin, target_count):
        """Resample the trajectories in a given bin.

        Parameters
        ----------
        bin : Bin
            Bin to resample.
        target_count : int
            Target number of trajectories for the bin.

        Returns
        -------
        Bin
            Resampled bin.

        """
        ...

    def __init__(
        self,
        seed=None,
        smallest_allowed_weight=1e-310,
        largest_allowed_weight=1.0,
    ):
        if not (0 < smallest_allowed_weight < 1):
            raise ValueError("'smallest_allowed_weight' must be between 0 and 1")
        if not (smallest_allowed_weight < largest_allowed_weight <= 1):
            raise ValueError("'largest_allowed_weight' must be between 'smallest_allowed_weight' and 1")

        seed = seed if seed is not None else secrets.randbits(128)
        logger.info(f'{seed=}')

        self.smallest_allowed_weight = smallest_allowed_weight
        self.largest_allowed_weight = largest_allowed_weight
        self.rng = np.random.default_rng(seed)

    def _split_by_threshold(self, bin):
        index = bin.bisect_weights(self.largest_allowed_weight, side='right')
        to_split = bin[index:]
        for segment in to_split:
            m = math.ceil(segment.weight / self.largest_allowed_weight)
            split(segment, bin, m=m)

    def _merge_by_threshold(self, bin):
        while True:
            index = bin.bisect_weights(self.smallest_allowed_weight)
            to_merge = bin[:index]
            if len(to_merge) < 2:
                return
            merge(to_merge, bin, rng=self.rng)

    def __call__(self, bin, target_count):
        total_weight = bin.weight

        bin = self.resample(bin, target_count)

        weights = bin.weights()
        if (weights <= 0).any():
            raise ConsistencyError('weights must be greater than 0')
        if not math.isclose(weights.sum(), total_weight, abs_tol=1e-12):
            raise ConsistencyError('resampling must preserve the total weight of the bin')

        self._split_by_threshold(bin)
        self._merge_by_threshold(bin)
        for segment in bin:
            if not (self.smallest_allowed_weight <= segment.weight <= self.largest_allowed_weight):
                logger.warning(
                    f'Unable to fulfill weight threshold conditions for {segment}. '
                    'The given threshold range is likely too small.'
                )

        return bin
