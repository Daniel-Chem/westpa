import abc
import logging
import secrets

import numpy as np

from .operations import split, merge

logger = logging.getLogger(__name__)


class ResamplingError(RuntimeError):
    pass


class Resampler(abc.ABC):
    """Defines a method for resampling an ensemble of segments.

    Parameters
    ----------
    seed : int, sequence of int, or numpy.random.Generator, optional
        Seed to initialize the pseudo-random number generator. Integer values
        must be non-negative.

    Attributes
    ----------
    rng : numpy.random.Generator

    """

    def __init__(self, *, seed=None):
        seed = seed if seed is not None else secrets.randbits(128)
        self._rng = np.random.default_rng(seed)
        logger.info(f'{seed=}')

    @property
    def rng(self):
        """Pseudo-random number generator."""
        return self._rng

    @staticmethod
    def split(segments, m=2):
        """Split a trajectory segment into two or more copies.

        Parameters
        ----------
        segment : Segment
            Segment to split.
        m : int, default 2
            Number of copies to create.

        Returns
        -------
        list of Segment
            Segments created by splitting `segment` into `m` equally weighted copies.

        """
        return split(segments, m=m)

    def merge(self, segments):
        """Merge multiple trajectory segments into a single segment. The
        surviving segment is chosen randomly according to weight.

        Parameters
        ----------
        segments : iterable of Segment
            Segments to merge.

        Returns
        -------
        Segment
            Copy of the surviving segment, assigned the total weight of
            `segments`.

        """
        return merge(segments, seed=self.rng)

    @abc.abstractmethod
    def resample(self, segments):
        """Resample an ensemble of segments."""
        ...

    def __call__(self, segments):
        # Calls a subclass's resample() method and performs consistency checks
        # (e.g., weights sum to 1) before returning the resampled segments.
        resampled_segments = self.resample(segments)

        weights = np.array([segment.weight for segment in resampled_segments])
        if (weights <= 0).any():
            raise ResamplingError('segment weights must be greater than 0')
        if (weights > 1).any():
            raise ResamplingError('segment weights must be less than or equal to 1')
        if not np.isclose(weights.sum(), 1):  # TODO: What should the tolerance be here?
            raise ResamplingError('segment weights must sum to 1')

        return resampled_segments
