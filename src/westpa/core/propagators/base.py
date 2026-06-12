import logging
import os
import secrets
import time
import traceback

import numpy as np

logger = logging.getLogger(__name__)


class Propagator:
    """Base class for propagators. Subclasses must override either the
    :meth:`propagate` method or the :meth:`propagate_block` method.

    Parameters
    ----------
    block_size : int, optional
        Block size (in number of segments) for propagation tasks.
        Defaults to 128 if :meth:`propagate_block` is overridden;
        otherwise defaults to 1.
    segment_dir_template : str, optional
        Template string specifying the directory in which to store output for a
        given segment. The string must contain ``{n_iter}`` and ``{seg_id}``
        replacement fields. If a relative path is provided, it is assumed to be
        relative to the current working directory. Defaults to
        ``'traj_segs/{n_iter:06d}/{seg_id:06d}'``.
    root_seed : int, optional
        Root random seed. Must be non-negative. The `root_seed` is combined
        with ``n_iter`` and ``seg_id`` values to create reproducible random
        seeds according to the approach described in
        `this section <https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds>`_
        of the NumPy reference manual. Defaults to ``secrets.randbits(128)``.
    bit_generator_type : type, optional
        Subclass of ``numpy.random.BitGenerator``. This parameter determines
        the type of bit generator passed to the :meth:`propagate` and
        :meth:`propagate_block` methods. Defaults to ``numpy.random.PCG64``.

    Attributes
    ----------
    block_size : int
    segment_dir_template : str
    root_seed : int
    bit_generator_type : type

    Methods
    -------
    propagate
    propagate_block
    make_segment_dir

    """

    DEFAULT_SEGMENT_DIR_TEMPLATE = 'traj_segs/{n_iter:06d}/{seg_id:06d}'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        a = cls.propagate is not Propagator.propagate
        b = cls.propagate_block is not Propagator.propagate_block
        if a == b:
            raise TypeError("subclasses of Propagator must override either 'propagate' or 'propagate_block'")

    def __init__(
        self,
        block_size=None,
        segment_dir_template=None,
        root_seed=None,
        bit_generator_type=None,
    ):
        if type(self) is Propagator:
            raise TypeError("Propagator can't be instantiated directly")

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

        if segment_dir_template is None:
            segment_dir_template = self.DEFAULT_SEGMENT_DIR_TEMPLATE
        else:
            try:
                segment_dir_template.format(n_iter=1, seg_id=0)
            except ValueError:
                raise

        if root_seed is None:
            root_seed = secrets.randbits(128)
        else:
            if not isinstance(root_seed, int):
                raise TypeError("'root_seed' must be an integer")
            if root_seed < 0:
                raise ValueError("'root_seed' must be non-negative")

        logger.info(f'{root_seed=}')

        if bit_generator_type is None:
            bit_generator_type = np.random.PCG64
        elif not issubclass(bit_generator_type, np.random.BitGenerator):
            raise TypeError("'bit_generator_type' must be a subclass of numpy.random.BitGenerator")

        self._block_size = block_size
        self._segment_dir_template = os.path.abspath(segment_dir_template)
        self._root_seed = root_seed
        self._bit_generator_type = bit_generator_type

    @property
    def block_size(self):
        """Block size for propagation tasks."""
        return self._block_size

    @property
    def segment_dir_template(self):
        """Segment directory template."""
        return self._segment_dir_template

    @property
    def root_seed(self):
        """Root seed integer."""
        return self._root_seed

    @property
    def bit_generator_type(self):
        """Bit generator type."""
        return self._bit_generator_type

    def propagate(self, segment, rng):
        """Propagate a single segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to propagate.
        rng : numpy.random.Generator
            Pseudo-random number generator for seeding the stochastic dynamics
            engine.

        Returns
        -------
        segment : :class:`Segment`
            Propagated segment.

        """
        return self.propagate_block([segment], rng)[0]

    def propagate_block(self, segments, rng):
        """Propagate a block of segments.

        Parameters
        ----------
        segments : sequence of :class:`Segment`
            Segments to propagate.
        rng : numpy.random.Generator
            Pseudo-random number generator for seeding the stochastic dynamics
            engine.

        Returns
        -------
        segments : sequence of :class:`Segment`
            Propagated segments.

        """
        return tuple(map(self.propagate, ((segment, rng) for segment in segments)))

    def make_segment_dir(self, segment):
        """Create a directory to store output for a given segment, and return
        its path.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to create output directory for.

        Returns
        -------
        segment_dir : str
            Absolute path of the new directory.

        """
        segment_dir = self.segment_dir_template.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        os.makedirs(segment_dir)
        return segment_dir

    def _get_rng(self, segment):
        seed = [segment.seg_id, segment.n_iter, self.root_seed]
        bit_generator = self._bit_generator_type(seed)
        return np.random.default_rng(bit_generator)

    def __call__(self, segments):
        # call propagate() if it is overridden; else call propagate_block()
        if type(self).propagate is not Propagator.propagate:
            for segment in segments:
                rng = self._get_rng(segment)
                start_walltime = time.perf_counter()
                try:
                    segment = self.propagate(segment, rng)
                except Exception:
                    segment.mark_as_failed(traceback.format_exc())
                else:
                    segment.walltime = time.perf_counter() - start_walltime
        else:
            rng = self._get_rng(segments[0])
            segments = self.propagate_block(segments, rng)

        return segments
