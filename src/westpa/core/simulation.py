import copy
import itertools
import logging
import math
import operator
import os
import time
from datetime import timedelta

import numpy as np
from sortedcontainers import SortedList

from .state import State
from .segment import Segment
from .propagators import Propagator, BatchedPropagator
from .binning import BinMapper, NopMapper
from .resamplers import HuberKimResampler, Resampler
from .source_sink import Source, Sink
from .plugins import Plugin
from .sim_manager import PropagationError
from ..work_managers import SerialWorkManager
from ..work_managers.core import WorkManager
from ._data_manager import DataManager

logger = logging.getLogger(__name__)


# Copied from https://docs.python.org/3/library/itertools.html#itertools.batched.
# TODO: Replace with itertools.batched() when we require Python >=3.12.
def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def trivial_pcoord_calculator(segment):
    """Set a segment's ``pcoord`` to ``final_state.coord``, and return the modified segment.

    Parameters
    ----------
    segment : Segment
        Segment with non-null value of ``final_state.coord``.

    Returns
    -------
    Segment
        Modified segment, with ``pcoord`` set to ``final_state.coord``.

    """
    segment.pcoord = segment.final_state.coord
    return segment


class Simulation:
    """Interface for initializing and running a weighted ensemble simulation.

    Parameters
    ----------
    datafile : str or BufferedIOBase
        HDF5 file used to store simulation data. Either a pathname (e.g.,
        ``'west.h5'``) or a binary stream may be provided.
    propagator : Propagator
        Routine for simulating dynamics for a fixed time interval :math:`\\tau`.
    pcoord_calculator : Callable[[Segment], Segment], optional
        Routine for computing the progress coordinate(s). It should take a
        propagated segment, set its ``pcoord`` attribute, and return the
        modified segment. Defaults to :obj:`~westpa.trivial_pcoord_calculator`,
        which sets ``pcoord`` to ``final_state.coord``.
    bin_mapper : BinMapper, optional
        Routine for grouping trajectories into bins. By default, all the
        trajectories are grouped into a single bin.
    bin_target_counts : int or sequence of int, default 1
        Target number of trajectories (allocation) for each bin. If passed an
        integer, the value will be applied to all the bins. If passed a
        sequence, its length must match ``bin_mapper.nbins``.
    resampler : Resampler, optional
        Routine for resampling the trajectories in each bin. Defaults to
        ``HuberKimResampler()``.
    source : Source, optional
        Distribution according to which to re-initiate (recycle) walkers that
        reach a sink. Must be provided together with `sinks`.
    sinks : Sink or iterable of Sink, optional
        One or more sink (target) regions. Must be provided together with `source`.
    istate_generator : Callable[[State], State], optional
        Routine for modifying the source distribution on the fly (e.g., by
        randomizing one or more degrees of freedom). It should take a state
        from `source` as input and return a new state.
    work_manager : WorkManager, optional
        Work manager for executing calls to `propagator`, `pcoord_calculator`, and
        `istate_generator`. By default, calls are executed serially.
    propagator_block_size : int, optional
        Number of segments to process in a given call to `propagator`. Defaults
        to 128 if `propagator` is a :class:`~westpa.BatchedPropagator`
        instance; otherwise defaults to 1.
    plugins : iterable of Plugin, optional
        One or more plugins to modify the simulation loop.

    Attributes
    ----------
    datafile : str
    propagator : Propagator
    pcoord_calculator : callable
    bin_mapper : BinMapper
    bin_target_counts : numpy.ndarray
    resampler : Resampler
    source : Source or None
    sinks : tuple of Sink
    work_manager : WorkManager
    propagator_block_size : int
    plugins : iterable of Plugin
    current_iteration : int

    Methods
    -------
    initialize
    run
    update_bins
    update_source_and_sinks
    add_plugin

    """

    def __init__(
        self,
        *,
        datafile,
        propagator,
        pcoord_calculator=None,
        bin_mapper=None,
        bin_target_counts=1,
        resampler=None,
        source=None,
        sinks=None,
        istate_generator=None,
        work_manager=None,
        propagator_block_size=None,
        plugins=None,
    ):
        self._propagator = None
        self._pcoord_calculator = None
        self._bin_mapper = None
        self._bin_target_counts = None
        self._resampler = None
        self._source = None
        self._sinks = ()
        self._istate_generator = None
        self._work_manager = None
        self._propagator_block_size = None
        self._plugins = SortedList(key=operator.attrgetter('priority'))

        self.data_manager = DataManager(datafile)

        self.propagator = propagator
        self.pcoord_calculator = pcoord_calculator or trivial_pcoord_calculator
        self.update_bins(bin_mapper or NopMapper(), bin_target_counts)
        self.resampler = resampler or HuberKimResampler()

        if source is not None or sinks is not None:
            if source is None or sinks is None:
                raise ValueError("'source' and 'sinks' must be provided together")
            if isinstance(sinks, Sink):
                sinks = [sinks]
            self.update_source_and_sinks(source, sinks)

        self.istate_generator = istate_generator
        self.work_manager = work_manager or SerialWorkManager()

        if propagator_block_size is None:
            if isinstance(self.propagator, BatchedPropagator):
                self.propagator_block_size = 128
            else:
                self.propagator_block_size = 1
        else:
            self.propagator_block_size = propagator_block_size

        for plugin in plugins or []:
            self.add_plugin(plugin)

        self.segments = []
        self.resampled_segments = []  # populated by _run_we()
        self.next_iter_segments = []  # populated by _prepare_new_iteration()

    @property
    def datafile(self):
        """HDF5 data file."""
        return self.data_manager.we_h5filename

    @property
    def propagator(self):
        """Propagator."""
        return self._propagator

    @propagator.setter
    def propagator(self, value):
        if not isinstance(value, Propagator):
            raise TypeError("'propagator' must be a Propagator object")
        self._propagator = value

    @property
    def pcoord_calculator(self):
        """Progress coordinate calculator."""
        return self._pcoord_calculator

    @pcoord_calculator.setter
    def pcoord_calculator(self, value):
        if not callable(value):
            raise TypeError("'pcoord_calculator' must be callable")
        self._pcoord_calculator = value

    @property
    def bin_mapper(self):
        """Bin mapper."""
        return self._bin_mapper

    @property
    def bin_target_counts(self):
        """Target number of trajectories for each bin."""
        return self._bin_target_counts

    @property
    def resampler(self):
        """Resampler."""
        return self._resampler

    @resampler.setter
    def resampler(self, value):
        if not isinstance(value, Resampler):
            raise TypeError("'resampler' must be a Resampler object")
        self._resampler = value

    @property
    def source(self):
        """Source distribution."""
        return self._source

    @property
    def sinks(self):
        """Sink (target) regions."""
        return self._sinks

    @property
    def istate_generator(self):
        """Initial state generator."""
        return self._istate_generator

    @istate_generator.setter
    def istate_generator(self, value):
        if value is not None and not callable(value):
            raise TypeError("'istate_generator' must be callable")
        self._istate_generator = value

    @property
    def work_manager(self):
        """Work manager."""
        return self._work_manager

    @work_manager.setter
    def work_manager(self, value):
        if not isinstance(value, WorkManager):
            raise TypeError("'work_manager' must be a WorkManager object")
        self._work_manager = value

    @property
    def propagator_block_size(self):
        """Batch size for propagation tasks."""
        return self._propagator_block_size

    @propagator_block_size.setter
    def propagator_block_size(self, value):
        if not isinstance(value, int):
            raise TypeError("'propagator_block_size' must be an integer")
        elif value < 1:
            raise ValueError("'propagator_block_size' must be at least 1")
        self._propagator_block_size = value

    @property
    def plugins(self):
        """Plugins."""
        return self._plugins

    @property
    def current_iteration(self):
        """Current iteration number."""
        if self.data_manager.we_h5file is None:
            reclose = True
            self.data_manager.open_backing(mode='r')
        else:
            reclose = False
        n = self.data_manager.current_iteration
        if reclose:
            self.data_manager.close_backing()
        return n

    @property
    def incomplete_segments(self):
        for segment in self.segments:
            if segment.status != Segment.Status.COMPLETE:
                yield segment

    def initialize(
        self,
        initial_states,
        weights=None,
    ):
        """Initialize the simulation.

        Parameters
        ----------
        initial_states : State or iterable of State
            States from which to initiate trajectories (one per trajectory).
        weights : 1-D array-like, optional
            Weight to assign each trajectory. Defaults to a uniform distribution.

        """
        if os.path.exists(self.datafile):
            reason = f'file {self.datafile!r} already exists'
            raise FileExistsError(f"can't initialize the simulation: {reason}")

        self.data_manager.prepare_backing()
        logger.info(f'Created HDF5 file {self.datafile!r}')

        if isinstance(initial_states, State):
            initial_states = [initial_states]
        else:
            initial_states = list(initial_states)

        if weights is None:
            weights = np.ones(len(initial_states))
        else:
            weights = np.array(weights, dtype=float)
            if len(weights) != len(initial_states):
                raise ValueError("length of 'weights' must match number of initial states")
        weights /= weights.sum()

        self.segments = [
            Segment(
                n_iter=1,
                seg_id=index,
                weight=weight,
                parent_id=-(1 + index),
                wtg_parent_ids={-(1 + index)},
                initial_state=state,
                status=Segment.Status.PREPARED,
            )
            for index, (state, weight) in enumerate(zip(initial_states, weights))
        ]

        self.data_manager.prepare_iteration(n_iter=1, segments=self.segments)
        self.data_manager.current_iteration = 1

        logger.info('Simulation prepared.')
        self._report_statistics(save_summary=True)
        self.data_manager.flush_backing()
        self.data_manager.close_backing()

    def run(self, n_iters=1, max_walltime=None):
        """Run the simulation.

        Parameters
        ----------
        n_iters : int, default 1
            Number of iterations to run.
        max_walltime : float, optional
            Maximum wall-clock time in seconds. If provided, the simulation
            will be terminated early if it is estimated that the next
            iteration would cause the runtime to exceed `max_walltime`.

        """
        with self.work_manager as work_manager:
            if work_manager.is_master:
                work_manager.install_sigint_handler()
                self._run(n_iters, max_walltime)
            else:
                work_manager.run()

    def update_bins(self, mapper, target_counts):
        """Update the bin mapper and target counts.

        Parameters
        ----------
        mapper : BinMapper
            Routine for grouping trajectory segments into bins.
        target_counts : int or sequence of int
            Target number of trajectories for each bin. If passed an integer,
            the value will be applied to all the bins. If passed a sequence,
            its length must match ``bin_mapper.nbins``.

        """
        if not isinstance(mapper, BinMapper):
            raise TypeError("'mapper' must be a BinMapper object")

        if isinstance(target_counts, int):
            target_counts = np.repeat(target_counts, mapper.nbins)
        else:
            target_counts = np.asarray(target_counts, dtype=int)
            if not len(target_counts) == mapper.nbins:
                raise ValueError("length of 'target_counts' must equal the number of bins")
        if (target_counts <= 0).any():
            raise ValueError("'target_counts' must be positive")

        self._bin_mapper = mapper
        self._bin_target_counts = target_counts

    def update_source_and_sinks(self, source, sinks):
        """Update the source and sinks.

        Parameters
        ----------
        source : Source
            Source distribution.
        sinks : iterable of Sink
            Sink (target) regions.

        """
        if not isinstance(source, Source):
            raise TypeError("'source' must be a Source object")
        sinks = tuple(sinks)
        if not all(isinstance(item, Sink) for item in sinks):
            raise TypeError("items in 'sinks' must be Sink objects")
        self._source = source
        self._sinks = sinks

    def add_plugin(self, plugin):
        """Add a plugin to the simulation.

        Parameters
        ----------
        plugin : Plugin
            Plugin to add.

        """
        if not isinstance(plugin, Plugin):
            raise TypeError("'plugin' must be a Plugin object")
        self._plugins.add(plugin)

    def _run(self, n_iters, max_walltime):
        self._prepare_run()

        start_time = time.time()
        stop_time = None
        if max_walltime:
            stop_time = start_time + max_walltime
            logger.info(f'Maximum wallclock time: {timedelta(seconds=max_walltime or 0)}')

        max_iter = self.current_iteration + n_iters - 1

        iter_elapsed = 0
        while self.current_iteration <= max_iter:
            n_iter = self.current_iteration
            if max_walltime and time.time() + 1.1 * iter_elapsed >= stop_time:
                logger.info(f'Iteration {n_iter} would require more than the alloted time. Ending run.')
                return
            try:
                iter_start_time = time.time()

                logger.info(time.asctime())
                logger.info(f'Iteration {n_iter} (of {max_iter})')

                self._prepare_iteration()
                self._propagate()
                self._run_we()
                self._prepare_new_iteration()
                self._finalize_iteration()

                iter_elapsed = time.time() - iter_start_time
                cputime = sum(segment.cputime for segment in self.segments)
                iter_summary = self.data_manager.get_iter_summary()
                iter_summary['walltime'] += iter_elapsed
                iter_summary['cputime'] = cputime
                self.data_manager.update_iter_summary(iter_summary)

                walltime = float(iter_summary['walltime'])
                if not math.isnan(walltime):  # may give NaN if starting a truncated simulation
                    walltime = timedelta(seconds=walltime)
                if not math.isnan(cputime):
                    cputime = timedelta(seconds=cputime)
                logger.info(f'Iteration wallclock: {walltime}, cputime: {cputime}')

                # Advance to the next iteration.
                self.data_manager.current_iteration += 1
                self.segments = copy.copy(self.next_iter_segments)
                self.resampled_segments.clear()
                self.next_iter_segments.clear()
            finally:
                self.data_manager.flush_backing()

        self._finalize_run()

        logger.info(time.asctime())
        logger.info('WESTPA run complete.')

    def _report_statistics(self, save_summary=False):
        seg_probs = np.fromiter(
            map(operator.attrgetter('weight'), self.segments),
            dtype=float,
            count=len(self.segments),
        )
        norm = seg_probs.sum()

        min_seg_prob = seg_probs[seg_probs != 0].min()
        max_seg_prob = seg_probs.max()
        seg_drange = np.log(max_seg_prob / min_seg_prob)

        eps = np.finfo(float).eps

        logger.info(f'number of segments:             {len(self.segments)}')
        logger.info(f'minimum non-zero probability:   {min_seg_prob:g}')
        logger.info(f'maximum non-zero probability:   {max_seg_prob:g}')
        logger.info(f'probability dynamic range (kT): {seg_drange:g}')
        logger.info(f'norm = {norm:g}, error in norm = {norm - 1:g} ({(norm - 1) / eps:.2g} * eps)')

        if min_seg_prob < 1e-100:
            logger.warning(
                'Minimum segment weight is < 1e-100 and might not be physically relevant. '
                'Please reconsider your progress coordinate or binning scheme.'
            )

        if save_summary:
            iter_summary = self.data_manager.get_iter_summary()
            iter_summary['n_particles'] = len(self.segments)
            iter_summary['norm'] = norm
            iter_summary['min_seg_prob'] = min_seg_prob
            iter_summary['max_seg_prob'] = max_seg_prob
            if np.isnan(iter_summary['cputime']):
                iter_summary['cputime'] = 0.0
            if np.isnan(iter_summary['walltime']):
                iter_summary['walltime'] = 0.0
            self.data_manager.update_iter_summary(iter_summary)

    def _prepare_run(self):
        self.data_manager.prepare_run()
        self._call_plugin_method(Plugin.prepare_run)

    def _finalize_run(self):
        self._call_plugin_method(Plugin.finalize_run)
        self.data_manager.finalize_run()

    def _prepare_iteration(self):
        logger.debug(f'beginning iteration {self.current_iteration}')

        if self.segments is None:
            self.segments = self.data_manager.get_segments()
            logger.debug(f'loaded {len(self.segments)} segments')
        else:
            logger.debug(f'using {len(self.segments)} pre-existing segments')

        incomplete_segments = list(self.incomplete_segments)

        n_incomplete = len(incomplete_segments)
        n_complete = len(self.segments) - n_incomplete

        logger.debug(f'{n_complete} segments are complete; {n_incomplete} are incomplete')

        if len(incomplete_segments) == len(self.segments):
            logger.info(f'Beginning iteration {self.current_iteration}')
        elif incomplete_segments:
            logger.info(f'Continuing iteration {self.current_iteration}')

        logger.info(f'{n_incomplete} segments remain in iteration {self.current_iteration} ({len(self.segments)} total)')

        self._call_plugin_method(Plugin.prepare_iteration)

    def _finalize_iteration(self):
        logger.debug('finalizing iteration {:d}'.format(self.current_iteration))
        self.data_manager.finalize_iteration(self.current_iteration, self.segments)
        self._call_plugin_method(Plugin.finalize_iteration)
        logger.info("Iteration completed successfully")

    def _propagate(self):
        self._call_plugin_method(Plugin.pre_propagation)

        futures = set()
        istate_futures = set()
        propagator_futures = set()
        pcoord_futures = set()

        unprepared_segments = []
        prepared_segments = []

        for segment in self.segments:
            if segment.initial_state is None:
                unprepared_segments.append(segment)
            elif segment.final_state is None:
                prepared_segments.append(segment)
            elif segment.pcoord is None:
                # Immediately dispatch pending pcoord calculation tasks
                future = self.work_manager.submit(self.pcoord_calculator, args=(segment,))
                pcoord_futures.add(future)
                futures.add(future)

        n_incomplete = len(unprepared_segments) + len(prepared_segments)
        logger.debug(f'iteration {self.current_iteration}: propagating {n_incomplete} segments')

        if unprepared_segments:
            states = self.source.random_choice(len(unprepared_segments), seed=self.resampler.rng)
            if self.istate_generator is not None:
                for state in states:
                    logger.debug(f'generating new initial state from source state {state}')
                    future = self.work_manager.submit(self.istate_generator, args=(state,))
                    istate_futures.add(future)
                    futures.add(future)
            else:
                for state in states:
                    logger.debug(f'using source state {state} directly')
                    segment = unprepared_segments.pop()
                    segment = segment.copy(initial_state=state, status=Segment.Status.PREPARED)
                    prepared_segments.append(segment)
                self.data_manager.write_initial_states(self.current_iteration, prepared_segments)

        for segments in batched(prepared_segments, self.propagator_block_size):
            future = self.work_manager.submit(self.propagator, args=(segments,))
            propagator_futures.add(future)
            futures.add(future)
        prepared_segments.clear()

        logger.info('Waiting for segments to complete...')

        while futures:
            future = self.work_manager.wait_any(futures)
            futures.remove(future)

            if future in istate_futures:
                istate_futures.remove(future)
                state = future.get_result()

                segment = unprepared_segments.pop()
                segment = segment.copy(initial_state=state, status=Segment.Status.PREPARED)
                prepared_segments.append(segment)

                self.segments[segment.seg_id] = segment
                self.data_manager.write_initial_states(self.current_iteration, segments=[segment])

                if len(prepared_segments) == self.propagator_block_size or not istate_futures:
                    future = self.work_manager.submit(self.propagator, args=(prepared_segments,))
                    propagator_futures.add(future)
                    futures.add(future)
                    prepared_segments.clear()

            elif future in propagator_futures:
                propagator_futures.remove(future)
                segments = future.get_result()

                for segment in segments:
                    if segment.status != Segment.Status.COMPLETE:
                        logger.error(f'propagation failed for segment {segment.seg_id}')
                        raise PropagationError(f'seg_id: {segment.seg_id}, reason: {segment.failure_reason}')
                    self.segments[segment.seg_id] = segment
                self.data_manager.write_final_states(self.current_iteration, segments)

                for segment in segments:
                    future = self.work_manager.submit(self.pcoord_calculator, args=(segment,))
                    pcoord_futures.add(future)
                    futures.add(future)

            elif future in pcoord_futures:
                pcoord_futures.remove(future)
                segment = future.get_result()
                self.segments[segment.seg_id] = segment
                self.data_manager.write_pcoords(self.current_iteration, segments=[segment])

        logger.debug('done with propagation')

        self.data_manager.flush_backing()
        self._call_plugin_method(Plugin.post_propagation)

    def _run_we(self):
        self._call_plugin_method(Plugin.pre_we)

        # Initialize the weight transfer graph.
        segments = [s.copy(wtg_parent_ids=[s.seg_id]) for s in self.segments]

        bins = self.bin_mapper.map(segments)
        for i, bin in enumerate(bins):
            if len(bin) == 0:
                continue
            target_count = self.bin_target_counts[i]
            bins[i] = self.resampler(bin, target_count)
        self.resampled_segments = list(itertools.chain(*bins))

        self._call_plugin_method(Plugin.post_we)

    def _prepare_new_iteration(self):
        recycled_segments = set()
        for sink in self.sinks:
            segments = {s for s in self.resampled_segments if s in sink}
            recycled_segments |= segments
            p = sum(s.weight for s in segments)
            logger.info(f'Recycled {p} probability ({len(segments)} walkers) from {sink!r}')

        for index, segment in enumerate(self.resampled_segments):
            parent = self.segments[segment.seg_id]

            if segment in recycled_segments:
                parent.endpoint_type = Segment.EndPointType.RECYCLED
                parent_id = -1
                initial_state = None
                status = None
            else:
                parent.endpoint_type = Segment.EndPointType.CONTINUES
                parent_id = parent.seg_id
                initial_state = parent.final_state
                status = Segment.Status.PREPARED

            new_segment = Segment(
                n_iter=segment.n_iter + 1,
                seg_id=index,
                weight=segment.weight,
                wtg_parent_ids=segment.wtg_parent_ids,
                parent_id=parent_id,
                initial_state=initial_state,
                status=status,
            )
            self.next_iter_segments.append(new_segment)

        for segment in self.segments:
            if segment.endpoint_type == Segment.EndPointType.UNSET:
                segment.endpoint_type = Segment.EndPointType.MERGED

        self.data_manager.prepare_iteration(self.current_iteration + 1, self.next_iter_segments)

        self._call_plugin_method(Plugin.prepare_new_iteration)

    def _call_plugin_method(self, base_method):
        for plugin in self._plugins:
            method = getattr(plugin.__class__, base_method.__name__)
            if method is not base_method:
                method(plugin, self)
