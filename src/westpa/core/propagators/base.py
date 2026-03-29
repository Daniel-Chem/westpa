import logging
import secrets

logger = logging.getLogger(__name__)


class Propagator:
    """Base class for propagators. Subclasses must override either the
    :meth:`propagate` method or the :meth:`propagate_block` method.

    Parameters
    ----------
    seed : int, optional
        Random seed. Must be non-negative. Defaults to ``secrets.randbits(128)``.
    block_size : int, optional
        Block size (in number of segments) for propagation tasks.
        Defaults to 128 if :meth:`propagate_block` is overridden;
        otherwise defaults to 1.

    Attributes
    ----------
    seed : int
    block_size : int

    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        a = cls.propagate is not Propagator.propagate
        b = cls.propagate_block is not Propagator.propagate_block
        if a == b:
            raise TypeError("subclasses of Propagator must override either 'propagate' or 'propagate_block'")

    def __init__(self, *, seed=None, block_size=None):
        if type(self) is Propagator:
            raise TypeError("Propagator can't be instantiated directly")

        if seed is None:
            seed = secrets.randbits(128)
        else:
            if not isinstance(seed, int):
                raise TypeError("'seed' must be an integer")
            if seed < 0:
                raise ValueError("'seed' must be non-negative")

        if block_size is None:
            if type(self).propagate_block is not Propagator.propagate_block:
                block_size = 128
            else:
                block_size = 1
        else:
            if not isinstance(block_size, int):
                raise TypeError("'block_size' must be an integer")
            if block_size < 1:
                raise ValueError("'block_size' must be positive")

        logger.info(f'{seed=}')

        self._seed = seed
        self._block_size = block_size

    @property
    def seed(self):
        """Random seed."""
        return self._seed

    @property
    def block_size(self):
        """Block size for propagation tasks."""
        return self._block_size

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
        return self.propagate_block([segment])

    def propagate_block(self, segments):
        """Propagate a block of segments.

        Parameters
        ----------
        segments : iterable of Segment
            Segments to propagate.

        Returns
        -------
        iterable of Segment
            Propagated segments.

        """
        return map(self.propagate, segments)

    def __call__(self, segments):
        return list(self.propagate_block(segments))
