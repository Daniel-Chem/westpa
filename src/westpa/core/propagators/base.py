import abc
import logging
import secrets

logger = logging.getLogger(__name__)


class Propagator(abc.ABC):
    """Defines a method for propagating a single segment.

    Parameters
    ----------
    seed : int, optional
        Random seed. Must be non-negative.

    Attributes
    ----------
    seed : int

    """

    def __init__(self, *, seed=None):
        self.seed = seed

    @property
    def seed(self):
        """Random seed."""
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is None:
            seed = secrets.randbits(128)
        else:
            if not isinstance(value, int):
                raise TypeError("'seed' must be an integer")
            if value < 0:
                raise ValueError("'seed' must be non-negative")
            seed = value
        logger.info(f'{seed=}')
        self._seed = seed

    @abc.abstractmethod
    def propagate(self, segment):
        """Propagate a single segment.

        Parameters
        ----------
        segment : Segment
            Segment to propagate.

        Returns
        -------
        Segment
            Propagated segment.

        """
        ...

    def __call__(self, segments):
        return [self.propagate(segment) for segment in segments]


class BatchedPropagator(Propagator):
    """Defines a method for propagating a batch of segments concurrently."""

    @abc.abstractmethod
    def propagate_batch(self, segments):
        """Propagate a batch of segments.

        Parameters
        ----------
        segments : iterable of Segment
            Segments to propagate.

        Returns
        -------
        iterable of Segment
            Propagated segments.

        """
        ...

    def propagate(self, segment):
        return self.propagate_batch([segment])[0]

    def __call__(self, segments):
        return list(self.propagate_batch(segments))
