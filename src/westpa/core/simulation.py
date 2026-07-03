import copy
import functools
import io
import itertools
import logging
import math
import operator
import os
import time
from collections.abc import Mapping
from datetime import timedelta

import numpy as np
from sortedcontainers import SortedList

from .state import State
from .segment import Segment
from .propagators import Propagator
from .binning import BinMapper, NopMapper
from .resamplers import HuberKimResampler, Resampler
from .source_sink import Source, Sink
from .plugins import Plugin
from .sim_manager import PropagationError
from ..work_managers import SerialWorkManager
from ..work_managers.core import WorkManager
from ._data_manager import DataManager

logger = logging.getLogger(__name__)


# copied from https://docs.python.org/3/library/itertools.html#itertools.batched
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


# decorator for methods that require initialization
def requires_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            raise RuntimeError(f'simulation must be initialized before calling {func.__name__!r}')
        return func(self, *args, **kwargs)

    return wrapper


def default_pcoord(segment):
    state = segment.final_state
    if state.coord is None:
        raise ValueError(f"{state} doesn't have a 'coord' value")
    return state.coord[np.newaxis, :]


class Simulation:
    """Interface for initializing and running a weighted ensemble (WE) simulation.

    Parameters
    ----------
    datafile : str or io.BytesIO
        HDF5 file used to store simulation data. Either a pathname (e.g.,
        ``'west.h5'``) or an in-memory stream may be provided.
    propagator : :class:`Propagator` or None
        Routine for propagating trajectories forward in time. Required when
        using the :meth:`run` method to run the simulation. If `propagator` is
        None, the simulation may be run in "WE-only" mode using the
        :meth:`get_segments`, :meth:`update_segments`, and
        :meth:`next_iteration` methods.
    pcoord_calculator : Callable[[:class:`Segment`], array_like] or Callable[[:class:`Segment`], Mapping[str, array_like]], optional
        Function that computes the progress coordinate(s) for a given
        trajectory segment. If the function returns a mapping, the ``'pcoord'``
        item is used as the progress coordinate; other items are stored as
        auxiliary data. If `pcoord_calculator` is not provided, the default
        progress coordinate will be used, equivalent to passing
        ``lambda seg: [seg.final_state.coord]``.
    bin_mapper : BinMapper, optional
        Routine for grouping trajectories into bins. By default, all the
        trajectories are grouped into a single bin.
    bin_target_counts : int or iterable of int, default 1
        Target number of trajectories (allocation) for each bin. If an integer
        is provided, the value will be applied to all the bins. If an iterable
        is provided, its length must match the ``nbins`` attribute of `bin_mapper`.
    resampler : :class:`Resampler`, optional
        Routine for resampling the trajectories in each bin. Defaults to
        :class:`HuberKimResampler()`.
    source : :class:`Source`, optional
        Distribution according to which to reinitiate (recycle) trajectories
        that reach a sink. Must be provided together with `sinks`.
    sinks : :class:`Sink` or iterable of :class:`Sink`, optional
        Sink (target) sets. Must be provided together with `source`.
    istate_generator : Callable[[:class:`State`], :class:`State`], optional
        Routine for modifying the source distribution on the fly. It should
        take a state from `source` as input and return a new state.
    work_manager : `WorkManager <work_managers.html>`_, optional
        Work manager for executing calls to `propagator`, `pcoord_calculator`, and
        `istate_generator`. By default, calls are executed serially.
    plugins : iterable of :class:`Plugin`, optional
        Plugins to execute at specific points (hooks) in the simulation loop.

    Attributes
    ----------
    datafile : str
    propagator : :class:`Propagator` or None
    pcoord_calculator : callable or None
    bin_mapper : :class:`BinMapper`
    bin_target_counts : numpy.ndarray
    resampler : :class:`Resampler`
    source : :class:`Source` or None
    sinks : tuple of :class:`Sink`
    work_manager : WorkManager
    plugins : sequence of :class:`Plugin`
    current_iteration : int or None
    initialized : bool

    Methods
    -------
    initialize
    run
    update_bins
    update_source_and_sinks
    add_plugin
    prepare_run
    finalize_run
    get_segments
    update_segments
    next_iteration

    """

    def __init__(
        self,
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
        self._plugins = SortedList(key=operator.attrgetter('priority'))

        self._data_manager = DataManager(datafile)
        self._n_iter = None
        self._segments = []
        self._resampled_segments = []  # populated by _run_we()
        self._next_iter_segments = []  # populated by _prepare_new_iteration()

        self.propagator = propagator
        self.pcoord_calculator = pcoord_calculator
        self.update_bins(bin_mapper or NopMapper(), bin_target_counts)
        self.resampler = resampler or HuberKimResampler()
        self.update_source_and_sinks(source, sinks)
        self.istate_generator = istate_generator

        self.work_manager = work_manager or SerialWorkManager()

        for plugin in plugins or []:
            self.add_plugin(plugin)

        initialized = False
        if isinstance(datafile, io.BytesIO):
            if datafile.getbuffer().nbytes > 0:
                initialized = True
        elif os.path.exists(datafile):
            initialized = True

        if initialized:
            logger.debug('opening existing simulation')
            self._data_manager.open_backing()
            self._n_iter = self._data_manager.current_iteration
            self._segments = self._data_manager.get_segments()
            logger.debug('iteration %d; loaded %d segments', self._n_iter, len(self._segments))
            self._data_manager.close_backing()

    @property
    def datafile(self):
        """HDF5 data file."""
        return self._data_manager.we_h5filename

    @property
    def propagator(self):
        """Propagator."""
        return self._propagator

    @propagator.setter
    def propagator(self, value):
        if value is not None and not isinstance(value, Propagator):
            raise TypeError("'propagator' must be a Propagator object or None")
        self._propagator = value

    @property
    def pcoord_calculator(self):
        """Progress coordinate calculator."""
        return self._pcoord_calculator

    @pcoord_calculator.setter
    def pcoord_calculator(self, value):
        if value is not None and not callable(value):
            raise TypeError("'pcoord_calculator' must be callable or None")
        self._pcoord_calculator = value

    @property
    def bin_mapper(self):
        """Bin mapper."""
        return self._bin_mapper

    @property
    def bin_target_counts(self):
        """Bin allocations."""
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
        """Sink sets."""
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
    def plugins(self):
        """Plugins in order of priority."""
        return self._plugins

    @property
    def current_iteration(self):
        """Current iteration number."""
        return self._n_iter

    @property
    def initialized(self):
        """Whether the simulation has been initialized."""
        return self._n_iter is not None

    def initialize(
        self,
        states,
        weights=None,
    ):
        """Initialize the simulation.

        Parameters
        ----------
        states : State or iterable of State
            States from which to initiate trajectories (one per trajectory).
        weights : 1-D array-like, optional
            Weight to assign each trajectory. Defaults to a uniform distribution.

        """
        if self.initialized:
            raise RuntimeError("can't initialize the simulation: already initialized")

        self._data_manager.prepare_backing()
        logger.info(f'Created HDF5 file {self.datafile!r}')

        if isinstance(states, State):
            states = [states]
        else:
            states = list(states)

        if weights is None:
            weights = np.ones(len(states))
        else:
            weights = np.array(weights, dtype=float)
            if len(weights) != len(states):
                raise ValueError("length of 'weights' must match number of initial states")
        weights /= weights.sum()

        self._segments = [
            Segment(
                n_iter=1,
                seg_id=index,
                weight=weight,
                parent_id=-(1 + index),
                wtg_parent_ids={-(1 + index)},
                initial_state=state,
                status=Segment.Status.PREPARED,
            )
            for index, (state, weight) in enumerate(zip(states, weights))
        ]

        self._data_manager.prepare_iteration(n_iter=1, segments=self._segments)
        self._data_manager.current_iteration = 1
        self._n_iter = 1

        logger.info('Simulation prepared.')
        self._report_statistics(save_summary=True)
        self._data_manager.flush_backing()
        self._data_manager.close_backing()

    @requires_initialization
    def run(self, n_iters=1, max_walltime=None):
        """Run the simulation.

        Parameters
        ----------
        n_iters : int, default 1
            Number of iterations to run.
        max_walltime : float, optional
            Maximum wall-clock time in seconds. If provided, the simulation
            will be stopped if it is estimated that the next iteration would
            cause the total runtime to exceed `max_walltime`.

        """
        if self.propagator is None:
            raise RuntimeError("'propagator' is required when using the run() method")

        with self.work_manager as work_manager:
            if work_manager.is_master:
                work_manager.install_sigint_handler()
                self._run(n_iters, max_walltime)
            else:
                work_manager.run()

    def update_bins(self, mapper, target_counts):
        """Update the bin mapper and allocations.

        Parameters
        ----------
        mapper : :class:`BinMapper`
            Routine for grouping trajectory segments into bins.
        target_counts : int or iterable of int
            Target number of trajectories for each bin. If an iterable is
            provided, its length must match ``bin_mapper.nbins``.

        """
        if not isinstance(mapper, BinMapper):
            raise TypeError("'mapper' must be a BinMapper object")

        if isinstance(target_counts, int):
            target_counts = np.repeat(target_counts, mapper.nbins)
        else:
            target_counts = np.fromiter(target_counts, dtype=int)
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
        source : :class:`Source` or None
            Source distribution. If None, `sinks` must also be None.
        sinks : :class:`Sink`, iterable of :class:`Sink`, or None
            One or more sink (target) sets. If None, `source` must also be None.

        """
        if not (source or sinks):
            self._source = None
            self._sinks = ()
            return

        if not (source and sinks):
            raise ValueError("'source' and 'sinks' must be provided together")
        if not isinstance(source, Source):
            raise TypeError("'source' must be a Source object")
        if isinstance(sinks, Sink):
            sinks = (sinks,)
        else:
            sinks = tuple(sinks)
            if not all(isinstance(item, Sink) for item in sinks):
                raise TypeError("'sinks' must be a Sink object or an iterable of Sink objects")

        self._source = source
        self._sinks = sinks

    def add_plugin(self, plugin):
        """Add a plugin to the simulation.

        Parameters
        ----------
        plugin : :class:`Plugin`
            Plugin to add.

        """
        if not isinstance(plugin, Plugin):
            raise TypeError("'plugin' must be a Plugin object")
        self._plugins.add(plugin)

    @requires_initialization
    def get_segments(self):
        """Retrieve the current segments.

        Returns
        -------
        segments : iterable of :class:`Segment`
            Current segments.

        """
        return self._segments

    @requires_initialization
    def update_segments(self, *segments):
        """Update the current segments.

        Parameters
        ----------
        *segments : :class:`Segment`
            One or more modified segments.

        """
        segments = np.array(segments, dtype=object)

        for segment in segments:
            if segment.n_iter != self._n_iter:
                raise ValueError(f"'n_iter' must match the current iteration ({self._n_iter})")
            if not (0 <= segment.seg_id < len(self._segments)):
                raise ValueError(f"unrecognized 'seg_id': {segment.seg_id}")

        has_initial_state = np.zeros(len(segments), dtype=bool)
        has_final_state = np.zeros(len(segments), dtype=bool)
        has_pcoord = np.zeros(len(segments), dtype=bool)

        for i, segment in enumerate(segments):
            if segment.initial_state is not None:
                has_initial_state[i] = True
            if segment.final_state is not None:
                has_final_state[i] = True
            if segment.pcoord is not None:
                has_pcoord[i] = True

        self._data_manager.update_seg_index(self._n_iter, segments)
        self._data_manager.write_auxdata(self._n_iter, segments)

        self._data_manager.write_initial_states(self._n_iter, segments[has_initial_state])
        self._data_manager.write_final_states(self._n_iter, segments[has_final_state])
        self._data_manager.write_pcoords(self._n_iter, segments[has_pcoord])

        for segment in segments:
            self._segments[segment.seg_id] = segment

    def next_iteration(self):
        """Resample the current segments and advance to the next iteration.

        In addition to resampling, this method is responsible for recycling any
        trajectories that have reached a sink. In contrast to earlier versions
        of WESTPA (``<= 2022``), recycling is done *after* resampling.

        """
        self._run_we()
        self._prepare_new_iteration()
        self._finalize_iteration()

        self._data_manager.current_iteration += 1
        self._n_iter += 1

        self._segments = copy.copy(self._next_iter_segments)
        self._resampled_segments.clear()
        self._next_iter_segments.clear()

    def _run(self, n_iters, max_walltime):
        self.prepare_run()

        start_time = time.time()
        stop_time = None
        if max_walltime:
            stop_time = start_time + max_walltime
            logger.info(f'Maximum wallclock time: {timedelta(seconds=max_walltime or 0)}')

        max_iter = self._n_iter + n_iters - 1

        iter_elapsed = 0
        while self._n_iter <= max_iter:
            if max_walltime and time.time() + 1.1 * iter_elapsed >= stop_time:
                logger.info(f'Iteration {self._n_iter} would require more than the alloted time. Ending run.')
                return
            try:
                iter_start_time = time.time()

                logger.info(time.asctime())
                logger.info(f'Iteration {self._n_iter} (of {max_iter})')

                self._prepare_iteration()
                self._propagate()

                iter_summary = self._data_manager.get_iter_summary()
                cputime = sum(segment.cputime for segment in self._segments)
                iter_summary['cputime'] = cputime

                self.next_iteration()

                iter_elapsed = time.time() - iter_start_time
                iter_summary['walltime'] += iter_elapsed
                self._data_manager.update_iter_summary(iter_summary)

                walltime = float(iter_summary['walltime'])
                if not math.isnan(walltime):  # may give NaN if starting a truncated simulation
                    walltime = timedelta(seconds=walltime)
                if not math.isnan(cputime):
                    cputime = timedelta(seconds=cputime)

                logger.info('Iteration completed successfully')
                logger.info(f'Iteration wallclock: {walltime}, cputime: {cputime}')
            finally:
                self._data_manager.flush_backing()

        self.finalize_run()

        logger.info(time.asctime())
        logger.info('WESTPA run complete.')

    def _report_statistics(self, save_summary=False):
        seg_probs = np.fromiter(
            map(operator.attrgetter('weight'), self._segments),
            dtype=float,
            count=len(self._segments),
        )
        norm = seg_probs.sum()

        min_seg_prob = seg_probs[seg_probs != 0].min()
        max_seg_prob = seg_probs.max()
        seg_drange = np.log(max_seg_prob / min_seg_prob)

        eps = np.finfo(float).eps

        logger.info(f'number of segments:             {len(self._segments)}')
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
            iter_summary = self._data_manager.get_iter_summary()
            iter_summary['n_particles'] = len(self._segments)
            iter_summary['norm'] = norm
            iter_summary['min_seg_prob'] = min_seg_prob
            iter_summary['max_seg_prob'] = max_seg_prob
            if np.isnan(iter_summary['cputime']):
                iter_summary['cputime'] = 0.0
            if np.isnan(iter_summary['walltime']):
                iter_summary['walltime'] = 0.0
            self._data_manager.update_iter_summary(iter_summary)

    def prepare_run(self):
        """Open the HDF5 file in read/write mode."""
        self._data_manager.prepare_run()
        self._call_plugin_hook(Plugin.prepare_run)

    def finalize_run(self):
        """Flush and close the HDF5 file."""
        self._call_plugin_hook(Plugin.finalize_run)
        self._data_manager.finalize_run()

    def _prepare_iteration(self):
        logger.debug('preparing iteration %d', self._n_iter)
        self._call_plugin_hook(Plugin.prepare_iteration)
        self._report_statistics(save_summary=True)

    def _finalize_iteration(self):
        logger.debug('finalizing iteration %d', self._n_iter)
        self._data_manager.finalize_iteration(self._n_iter, self._segments)

    def _calculate_pcoords(self, segments):
        future_map = {}
        if self.pcoord_calculator is not None:
            for segment in segments:
                future = self.work_manager.submit(self.pcoord_calculator, args=(segment,))
                future_map[future] = segment
        else:
            for segment in segments:
                segment.pcoord = default_pcoord(segment)
            self._data_manager.write_pcoords(self._n_iter, segments)
        return future_map

    # manages istate generation, propagation, and pcoord calculation tasks
    def _propagate(self):
        istate_futures = set()
        propagator_futures = set()
        pcoord_future_map = {}

        # partition segments by status
        unprepared_segments = []
        prepared_segments = []
        complete_segments = []
        for segment in self._segments:
            match segment.status:
                case Segment.Status.UNSET:
                    unprepared_segments.append(segment)
                case Segment.Status.PREPARED:
                    prepared_segments.append(segment)
                case Segment.Status.COMPLETE:
                    complete_segments.append(segment)

        n_incomplete = len(unprepared_segments) + len(prepared_segments)
        logger.debug('iteration %d: propagating %d segments', self._n_iter, n_incomplete)

        # dispatch pending istate generation tasks
        if unprepared_segments:
            states = self.source.random_choice(len(unprepared_segments), rng=self.resampler.rng)
            if self.istate_generator is not None:
                for state in states:
                    logger.debug('generating new initial state from %s', state)
                    future = self.work_manager.submit(self.istate_generator, args=(state,))
                    istate_futures.add(future)
            else:
                segments = []
                for state in states:
                    logger.debug('using %s directly as an initial state', state)
                    segment = unprepared_segments.pop()
                    segment.initial_state = state
                    segments.append(segment)
                self._data_manager.write_initial_states(self._n_iter, segments)
                prepared_segments += segments

        # dispatch pending propagation tasks
        for segments in batched(prepared_segments, self.propagator.block_size):
            future = self.work_manager.submit(self.propagator, args=(segments,))
            propagator_futures.add(future)
        prepared_segments.clear()

        # dispatch pending pcoord calculation tasks
        if segments := [s for s in complete_segments if s.pcoord is None]:
            pcoord_future_map |= self._calculate_pcoords(segments)

        logger.info('Waiting for segments to complete...')

        while futures := istate_futures | propagator_futures | pcoord_future_map.keys():
            future = self.work_manager.wait_any(futures)

            if future in istate_futures:
                istate_futures.remove(future)
                state = future.get_result()

                segment = unprepared_segments.pop()
                segment.initial_state = state
                prepared_segments.append(segment)

                self._segments[segment.seg_id] = segment
                self._data_manager.write_initial_states(self._n_iter, segments=[segment])

                if len(prepared_segments) == self.propagator.block_size or not istate_futures:
                    future = self.work_manager.submit(self.propagator, args=(prepared_segments,))
                    propagator_futures.add(future)
                    prepared_segments.clear()

            elif future in propagator_futures:
                propagator_futures.remove(future)
                segments = future.get_result()

                for segment in segments:
                    if segment.status != Segment.Status.COMPLETE:
                        logger.error('propagation failed for segment %d', segment.seg_id)
                        raise PropagationError(f'seg_id: {segment.seg_id}, reason: {segment.failure_reason}')
                    self._segments[segment.seg_id] = segment
                self._data_manager.write_final_states(self._n_iter, segments)

                pcoord_future_map |= self._calculate_pcoords(segments)

            elif future in pcoord_future_map:
                segment = pcoord_future_map.pop(future)
                result = future.get_result()

                if isinstance(result, Mapping):
                    if 'pcoord' not in result:
                        raise RuntimeError("'pcoord_calculator' return value must have a 'pcoord' key")
                    for key, value in result.items():
                        if key == 'pcoord':
                            segment.pcoord = value
                        else:
                            segment.data[key] = value
                else:
                    segment.pcoord = result

                self._segments[segment.seg_id] = segment
                self._data_manager.write_pcoords(self._n_iter, segments=[segment])

        logger.debug('done with propagation')
        self._data_manager.flush_backing()

    def _run_we(self):
        self._call_plugin_hook(Plugin.pre_we)

        # initialize the weight transfer graph
        segments = [s.replace(wtg_parent_ids=[s.seg_id]) for s in self._segments]

        bins = list(self.bin_mapper(segments))
        for i, bin in enumerate(bins):
            bins[i] = self.resampler(bin, target_count=self.bin_target_counts[i])

        self._resampled_segments = list(itertools.chain(*bins))

    def _prepare_new_iteration(self):
        recycled_segments = set()
        for index, sink in enumerate(self.sinks, start=1):
            if segments := set(filter(sink.indicator, self._resampled_segments)):
                recycled_segments |= segments
                p = sum(map(operator.attrgetter('weight'), segments))
                n = len(segments)
                label = sink.label if sink.label else index
                logger.info(f'Recycled {p} probability ({n} walkers) from sink {label!r}')

        for index, segment in enumerate(self._resampled_segments):
            parent = self._segments[segment.seg_id]

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
            self._next_iter_segments.append(new_segment)

        for segment in self._segments:
            if segment.endpoint_type == Segment.EndPointType.UNSET:
                segment.endpoint_type = Segment.EndPointType.MERGED

        self._data_manager.prepare_iteration(self._n_iter + 1, self._next_iter_segments)

    def _call_plugin_hook(self, base_method):
        for plugin in self._plugins:
            method = getattr(plugin.__class__, base_method.__name__)
            if method is not base_method:
                method(plugin, self)
