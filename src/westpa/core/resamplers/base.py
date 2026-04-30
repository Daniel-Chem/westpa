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
        Pseudo-random number generator to use. Defaults to NumPy's ``default_rng()``.

    Attributes
    ----------
    rng : numpy.random.Generator
        Pseudo-random number generator.

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

    def __init__(self, rng=None):
        if rng is None:
            seed = secrets.randbits(128)
            logger.info(f"Using NumPy's default random generator, {seed=}")
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(rng)

    def __call__(self, bin, target_count):
        total_weight = bin.weight

        bin = self.resample(bin, target_count)

        weights = bin.weights()
        if (weights <= 0).any():
            raise ConsistencyError('weights must be greater than 0')
        if not math.isclose(weights.sum(), total_weight, abs_tol=1e-12):
            raise ConsistencyError('resampling must preserve the total weight of the bin')

        return bin
