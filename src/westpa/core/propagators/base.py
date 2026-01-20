import inspect
import logging
import secrets

logger = logging.getLogger(__name__)


class Propagator:
    """Base class for propagators."""

    def __init__(self, *, seed=None):
        self.seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        seed = value if value is not None else secrets.randbits(128)
        logger.info(f'{seed=}')
        self._seed = seed

    def propagate(self, segment):
        """Propagate a single trajectory segment.

        Parameters
        ----------
        segment : Segment
            Segment to propagate.

        Returns
        -------
        Segment
            Propagated segment.

        """
        raise NotImplementedError

    def propagate_batch(self, segments):
        """Propagate a batch of trajectory segments.

        Parameters
        ----------
        segments : iterable of Segment
            Segments to propagate.

        Returns
        -------
        iterable of Segments
            Propagated segments.

        """
        return [self.propagate(segment) for segment in segments]

    def __call__(self, segments):
        return self.propagate_batch(segments)

    def __repr__(self):
        parameters = inspect.signature(self.__init__).parameters
        args = ', '.join(f'{name}={getattr(self, name)!r}' for name in parameters)
        return type(self).__name__ + '(' + args + ')'
