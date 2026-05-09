import abc
import logging
import math
import secrets

import numpy as np

from westpa.core.we_driver import ConsistencyError

logger = logging.getLogger(__name__)


class Resampler(abc.ABC):
    """Base class for resamplers. Subclasses must implement the :meth:`resample` method.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Pseudo-random number generator to use. Defaults to
        ``numpy.random.default_rng()``.
    smallest_allowed_weight : float, default 1e-310
        Minimum weight threshold.
    largest_allowed_weight : float, default 1.0
        Maximum weight threshold.
    thresholds : bool, default True
        Whether to enforce weight thresholds. If True, this is done after
        calling the subclass's :meth:`resample` method.

    Attributes
    ----------
    rng : numpy.random.Generator
        Pseudo-random number generator.
    smallest_allowed_weight : float
        Minimum weight threshold.
    largest_allowed_weight : float
        Maximum weight threshold.
    thresholds : bool
        Whether to enforce weight thresholds.

    Methods
    -------
    resample

    """

    def __init__(
        self,
        *,
        rng=None,
        smallest_allowed_weight=1e-310,
        largest_allowed_weight=1.0,
        thresholds=True,
    ):
        if rng is None:
            seed = secrets.randbits(128)
            self.rng = np.random.default_rng(seed)
            logger.info(f"Using NumPy default_rng({seed=})")
        else:
            self.rng = np.random.default_rng(rng)

        if not (0 < smallest_allowed_weight < 1):
            raise ValueError("'smallest_allowed_weight' must be between 0 and 1")
        if not (smallest_allowed_weight < largest_allowed_weight <= 1):
            raise ValueError("'largest_allowed_weight' must be between 'smallest_allowed_weight' and 1")
        self.smallest_allowed_weight = smallest_allowed_weight
        self.largest_allowed_weight = largest_allowed_weight

        self.thresholds = thresholds

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
        resampled_bin : Bin
            Resampled bin.

        """
        ...

    def _split_by_threshold(self, bin):
        index = bin.bisect_weights(self.largest_allowed_weight, side='right')
        to_split = bin[index:]
        for segment in to_split:
            m = math.ceil(segment.weight / self.largest_allowed_weight)
            bin.split(segment, m=m)

    def _merge_by_threshold(self, bin):
        while True:
            index = bin.bisect_weights(self.smallest_allowed_weight)
            to_merge = bin[:index]
            if len(to_merge) < 2:
                return
            bin.merge(to_merge, rng=self.rng)

    def __call__(self, bin, target_count):
        if len(bin) == 0:
            return bin

        bin_weight = bin.weight

        resampled_bin = self.resample(bin, target_count)

        weights = bin.weights()
        if (weights <= 0).any():
            raise ConsistencyError('weights must be greater than 0')
        if not math.isclose(weights.sum(), bin_weight, abs_tol=1e-12):
            raise ConsistencyError('resampling must preserve the total weight of the bin')

        if self.thresholds:
            self._split_by_threshold(resampled_bin)
            self._merge_by_threshold(resampled_bin)
            for segment in resampled_bin:
                if not (self.smallest_allowed_weight <= segment.weight <= self.largest_allowed_weight):
                    logger.warning(
                        'Unable to fulfill weight threshold conditions for %s. The given threshold range is likely too small.',
                        segment,
                    )

        return resampled_bin
