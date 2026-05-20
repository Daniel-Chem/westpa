import logging
import secrets
import time
import traceback

logger = logging.getLogger(__name__)


class Propagator:
    """Base class for propagators. Subclasses must override either the
    :meth:`propagate` method or the :meth:`propagate_block` method.

    Parameters
    ----------
    root_seed : int, optional
        Root seed integer to use in the :meth:`get_child_seed` method.
        Must be non-negative. Defaults to ``secrets.randbits(128)``.
    block_size : int, optional
        Block size (in number of segments) for propagation tasks.
        Defaults to 128 if :meth:`propagate_block` is overridden;
        otherwise defaults to 1.

    Attributes
    ----------
    root_seed : int
    block_size : int

    """

    DEFAULT_SEGMENT_DIR_TEMPLATE = 'traj_segs/{n_iter:06d}/{seg_id:06d}'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        a = cls.propagate is not Propagator.propagate
        b = cls.propagate_block is not Propagator.propagate_block
        if a == b:
            raise TypeError("subclasses of Propagator must override either 'propagate' or 'propagate_block'")

    def __init__(self, root_seed=None, block_size=None):
        if type(self) is Propagator:
            raise TypeError("Propagator can't be instantiated directly")

        if root_seed is None:
            root_seed = secrets.randbits(128)
        else:
            if not isinstance(root_seed, int):
                raise TypeError("'root_seed' must be an integer")
            if root_seed < 0:
                raise ValueError("'root_seed' must be non-negative")

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

        logger.info(f'{root_seed=}')

        self._root_seed = root_seed
        self._block_size = block_size

    @property
    def root_seed(self):
        """Root seed."""
        return self._root_seed

    @property
    def block_size(self):
        """Block size for propagation tasks."""
        return self._block_size

    def get_child_seed(self, segment):
        """Return a segment-specific seed for PRNG initialization.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment from which to derive the worker ID component of the child
            seed.

        Returns
        -------
        sequence of int
            The sequence ``[segment.seg_id, segment.n_iter, self.root_seed]``.
            For the rationale behind this choice, see
            `this section <https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds>`_
            of the NumPy reference manual.

        """
        return [segment.seg_id, segment.n_iter, self.root_seed]

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
        segments : sequence of Segment
            Segments to propagate.

        Returns
        -------
        iterable of Segment
            Propagated segments.

        """
        return map(self.propagate, segments)

    def __call__(self, segments):
        # call propagate() if it is overridden; else call propagate_block()
        if type(self).propagate is not Propagator.propagate:
            for segment in segments:
                start_walltime = time.perf_counter()
                start_cputime = time.process_time()
                try:
                    segment = self.propagate(segment)
                except Exception:
                    segment.mark_as_failed(traceback.format_exc())
                else:
                    segment.walltime = time.perf_counter() - start_walltime
                    segment.cputime = time.process_time() - start_cputime
        else:
            segments = tuple(self.propagate_block(segments))

        return segments
