import copy
import logging
import operator
import os
import time
from collections.abc import Container
from datetime import timedelta

import numpy as np

from .core.sim_manager import PropagationError
from .core.segment import Segment
from .core.we_driver import ConsistencyError
from .work_managers import SerialWorkManager
from .work_managers.core import WorkManager
from ._data_manager import DataManager
from ._state import State

logger = logging.getLogger(__name__)


class Simulation:
    """Interface for initializing and running a weighted ensemble simulation.

    Parameters
    ----------
    datafile : str
        Path to the HDF5 file used to store simulation data (e.g., ``'west.h5'``).
    propagator : Callable[[Segment], Segment]
        Routine that runs dynamics for a given trajectory segment. It should
        read a segment's ``initial_state``, set its ``final_state``, and return the
        modified segment.
    resampler : Callable[[Iterable[Segment]], Iterable[Segment]]
        Routine that takes a set of propagated trajectory segments, performs
        resampling (e.g., using the :func:`split` and :func:`merge` functions),
        and returns the resampled segments.
    pcoord_calculator : Callable[[Segment], Segment], optional
        Routine that computes the progress coordinate time series for a given
        trajectory segment. It should take a propagated segment, sets its
        ``pcoord`` attribute, and return the modified segment.
    source : Source, optional
        Source (initial) distribution according to which walkers that reach
        the `sink` are re-initiated (recycled). Must be provided together with
        `sink`.
    sink : Sink, optional
        Sink (target) region from which walkers are recycled according the
        `source` distribution. Must be provided together with `source`.
    work_manager : WorkManager, optional
        Work manager for executing calls to `propagator` and `pcoord_calculator`.
        By default, calls are executed serially.
        Some work managers (e.g., those based on Python's ``multiprocessing``
        package) may require `propagator` and `pcoord_calculator` to be
        `picklable <https://docs.python.org/3/library/pickle.html>`_.

    Attributes
    ----------
    datafile : str
    work_manager : WorkManager
    n_iter : int
    incomplete_segments : Iterator[Segment]

    Methods
    -------
    initialize
    run

    """

    def __init__(
        self,
        *,
        datafile,
        propagator,
        resampler,
        pcoord_calculator=None,
        source=None,
        sink=None,
        work_manager=None,
    ):
        self.data_manager = DataManager(datafile)

        self.propagator = propagator
        self.resampler = resampler
        self.pcoord_calculator = pcoord_calculator

        if source is not None or sink is not None:
            if source is None or sink is None:
                raise ValueError("'source' and 'sink' must be provided together")
        self.source = source
        self.sink = sink

        self.work_manager = work_manager or SerialWorkManager()

        self.current_iter_segments = []
        self.resampled_segments = []  # populated by _run_we()
        self.next_iter_segments = []  # populated by _prepare_next_iteration()

    @property
    def datafile(self):
        """HDF5 file used to store simulation data."""
        return self.data_manager.we_h5filename

    @property
    def work_manager(self):
        """Work manager for launching tasks."""
        return self._work_manager

    @work_manager.setter
    def work_manager(self, value):
        if not isinstance(value, WorkManager):
            raise TypeError("'work_manager' must be a WorkManager object")
        self._work_manager = value

    @property
    def n_iter(self):
        """Current iteration number."""
        return self.data_manager.current_iteration

    @property
    def incomplete_segments(self):
        """Incomplete segments remaining in the current iteration."""
        for segment in self.current_iter_segments:
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
            Microstates from which to start trajectories (one per trajectory).
        weights : 1-D array-like, optional
            Weight to assign each trajectory. By default, the trajectories are
            assigned equal weights.

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

        self.current_iter_segments = [
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

        self.data_manager.prepare_iteration(n_iter=1, segments=self.current_iter_segments)
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

    def _run(self, n_iters, max_walltime):
        self.data_manager.open_backing()

        start_time = time.time()
        if max_walltime:
            stop_time = start_time + max_walltime
            logger.info('Maximum wallclock time: %s' % timedelta(seconds=max_walltime or 0))
        else:
            stop_time = None

        max_iter = self.n_iter + n_iters - 1

        iter_walltime = 0
        while self.n_iter <= max_iter:
            if max_walltime and time.time() + 1.1 * iter_walltime >= stop_time:
                logger.info(f'Iteration {self.n_iter} would require more than the allotted time. Ending run.')
                return

            try:
                iter_start_time = time.time()

                logger.info('%s' % time.asctime())
                logger.info('Iteration %d (%d requested)' % (self.n_iter, max_iter))

                self._prepare_iteration()
                self._propagate()
                self._run_we()
                self._prepare_next_iteration()

                iter_walltime = time.time() - iter_start_time
                iter_summary = self.data_manager.get_iter_summary()
                iter_summary['walltime'] += iter_walltime
                iter_summary['cputime'] = sum(segment.cputime for segment in self.current_iter_segments)
                self.data_manager.update_iter_summary(iter_summary)

                # TODO: Clean this block up.
                try:
                    # This may give NaN if starting a truncated simulation
                    walltime = timedelta(seconds=float(iter_summary['walltime']))
                except ValueError:
                    walltime = 0.0
                try:
                    cputime = timedelta(seconds=float(iter_summary['cputime']))
                except ValueError:
                    cputime = 0.0
                logger.info(f'Iteration wallclock: {walltime}, cputime: {cputime}')

                self.data_manager.current_iteration += 1
                self.current_iter_segments = copy.copy(self.next_iter_segments)
                self.resampled_segments.clear()
                self.next_iter_segments.clear()

            finally:
                self.data_manager.flush_backing()

        self.data_manager.close_backing()

        logger.info('%s' % time.asctime())
        logger.info('WEST run complete.')

    def _report_statistics(self, save_summary=False):
        seg_probs = np.fromiter(
            map(operator.attrgetter('weight'), self.current_iter_segments),
            dtype=float,
            count=len(self.current_iter_segments),
        )
        norm = seg_probs.sum()

        min_seg_prob = seg_probs[seg_probs != 0].min()
        max_seg_prob = seg_probs.max()
        seg_drange = np.log(max_seg_prob / min_seg_prob)

        eps = np.finfo(float).eps

        logger.info(f'number of segments:             {len(self.current_iter_segments)}')
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
            iter_summary['n_particles'] = len(self.current_iter_segments)
            iter_summary['norm'] = norm
            iter_summary['min_seg_prob'] = min_seg_prob
            iter_summary['max_seg_prob'] = max_seg_prob
            if np.isnan(iter_summary['cputime']):
                iter_summary['cputime'] = 0.0
            if np.isnan(iter_summary['walltime']):
                iter_summary['walltime'] = 0.0
            self.data_manager.update_iter_summary(iter_summary)

    def _prepare_iteration(self):
        logger.debug('beginning iteration {:d}'.format(self.n_iter))

        if self.current_iter_segments is None:
            self.current_iter_segments = self.data_manager.get_segments()
            logger.debug(f'loaded {len(self.current_iter_segments)} segments')
        else:
            logger.debug(f'using {len(self.current_iter_segments)} pre-existing segments')

        incomplete_segments = list(self.incomplete_segments)

        n_incomplete = len(incomplete_segments)
        n_complete = len(self.current_iter_segments) - n_incomplete

        logger.debug(f'{n_complete} segments are complete; {n_incomplete} are incomplete')

        if len(incomplete_segments) == len(self.current_iter_segments):
            logger.info(f'Beginning iteration {self.n_iter}')
        elif incomplete_segments:
            logger.info(f'Continuing iteration {self.n_iter}')

        logger.info(f'{n_incomplete} segments remain in iteration {self.n_iter} ' f'({len(self.current_iter_segments)} total)')

    def _propagate(self):
        segments = list(self.incomplete_segments)
        logger.debug(f'iteration {self.n_iter}: propagating {len(segments)} segments')

        futures = set()
        propagator_futures = set()
        pcoord_futures = set()

        for segment in segments:
            future = self.work_manager.submit(self.propagator, args=(segment,))
            propagator_futures.add(future)
            futures.add(future)

        logger.info('Waiting for segments to complete...')

        while futures:
            future = self.work_manager.wait_any(futures)
            futures.remove(future)

            segment = future.get_result()

            if future in propagator_futures:
                propagator_futures.remove(future)

                if segment.status != Segment.Status.COMPLETE:
                    logger.error(f'propagation failed for segment {segment.seg_id}')
                    raise PropagationError(f'seg_id: {segment.seg_id}, reason: {segment.failure_reason}')

                self.current_iter_segments[segment.seg_id] = segment

                if self.pcoord_calculator is not None:
                    pcoord_future = self.work_manager.submit(self.pcoord_calculator, args=(segment,))
                    pcoord_futures.add(pcoord_future)
                    futures.add(pcoord_future)
                else:
                    self.data_manager.update_segments(self.n_iter, segments=[segment])

            elif future in pcoord_futures:
                pcoord_futures.remove(future)
                self.current_iter_segments[segment.seg_id] = segment
                self.data_manager.update_segments(self.n_iter, segments=[segment])

        logger.debug('done with propagation')

        self.data_manager.flush_backing()

    def _run_we(self):
        replicas = [
            Segment(
                n_iter=segment.n_iter,
                seg_id=segment.seg_id,
                weight=segment.weight,
                parent_id=segment.parent_id,
                wtg_parent_ids=[segment.seg_id],  # initialize weight transfer graph
                initial_state=segment.initial_state,
                final_state=segment.final_state,
                pcoord=segment.pcoord,
                data=segment.data,
            )
            for segment in self.current_iter_segments
        ]
        self.resampled_segments = list(self.resampler(replicas))

        weights = np.array([segment.weight for segment in self.resampled_segments])
        if (weights <= 0).any():
            raise ConsistencyError('segment weights must be greater than 0')
        if (weights > 1).any():
            raise ConsistencyError('segment weights must be less than or equal to 1')
        if not np.isclose(weights.sum(), 1):  # TODO: What should the tolerance be here?
            raise ConsistencyError('segment weights must sum to 1')

    def _prepare_next_iteration(self):
        if self.sink is not None:
            recycled_segments = set(filter(self.sink.indicator, self.resampled_segments))
            new_initial_states = self.source.random_sample(len(recycled_segments))
        else:
            recycled_segments = set()
            new_initial_states = []

        for index, segment in enumerate(self.resampled_segments):
            parent = self.current_iter_segments[segment.seg_id]

            if segment in recycled_segments:
                parent.endpoint_type = Segment.EndPointType.RECYCLED
                parent_id = -len(new_initial_states)
                initial_state = new_initial_states.pop()
            else:
                parent.endpoint_type = Segment.EndPointType.CONTINUES
                parent_id = parent.seg_id
                initial_state = parent.final_state

            new_segment = Segment(
                n_iter=segment.n_iter + 1,
                seg_id=index,
                weight=segment.weight,
                parent_id=parent_id,
                wtg_parent_ids=segment.wtg_parent_ids,
                initial_state=initial_state,
                status=Segment.Status.PREPARED,
            )
            self.next_iter_segments.append(new_segment)

        for segment in self.current_iter_segments:
            if segment.endpoint_type == Segment.EndPointType.UNSET:
                segment.endpoint_type = Segment.EndPointType.MERGED

        self.data_manager.update_segments(self.n_iter, self.current_iter_segments)
        self.data_manager.prepare_iteration(self.n_iter + 1, self.next_iter_segments)


class Source:
    """Represents a source (initial) distribution.

    Parameters
    ----------
    states : State or iterable of State
        One or more source states.
    p : 1-D array-like, optional
        Probability to assign each state. Defaults to a uniform distribution.

    Attributes
    ----------
    states : sequence of State
        Source states.
    p : numpy.ndarray
        Probability assigned to each state.

    Examples
    --------
    Single state:

    >>> import westpa
    >>> westpa.Source(westpa.State(label='a'))
    Source([State(label='a')], p=[1.0])

    Two states with equal probabilities:

    >>> states = [westpa.State(label='a'), westpa.State(label='b')]
    >>> westpa.Source(states)
    Source([State(label='a'), State(label='b')], p=[0.5, 0.5])

    Two states with different probabilities:

    >>> westpa.Source(states, p=[0.7, 0.3])
    Source([State(label='a'), State(label='b')], p=[0.7, 0.3])


    """

    def __init__(self, states, p=None):
        if isinstance(states, State):
            states = [states]
        else:
            states = list(states)
            if not all(isinstance(item, State) for item in states):
                raise TypeError("items in 'states' must be westpa.State objects")

        if p is None:
            p = np.ones(len(states))
        else:
            p = np.asarray(p, dtype=float)
            if len(p) != len(states):
                raise ValueError("length of 'p' must match number of states")
        p /= p.sum()

        self.states = states
        self.p = p

    def __repr__(self):
        args = f'{self.states}, p={self.p.tolist()}'
        return type(self).__name__ + '(' + args + ')'

    def random_sample(self, k, seed=None):
        rng = np.random.default_rng(seed)
        return rng.choice(self.states, p=self.p, size=k).tolist()


class Sink(Container):
    """Represents a sink (target) region.

    Parameters
    ----------
    indicator : Callable[[Segment], bool]
        Function that returns True if a given trajectory segment reached the
        sink, False otherwise. This function may assume that the segment has
        completed propagation and that its ``pcoord`` attribute has been set.

    Attributes
    ----------
    indicator : Callable[[Segment], bool]
        Indicator function.

    Methods
    -------
    __contains__

    Examples
    --------
    Create a sink containing segments with final (``-1``), first-dimension
    (``0``) progress coordinate values greater than one:

    >>> import westpa
    >>> sink = westpa.Sink(lambda segment: segment.pcoord[-1, 0] > 1.0)

    Test for membership:

    >>> westpa.Segment(pcoord=[[0.9]]) in sink
    False
    >>> westpa.Segment(pcoord=[[1.1]]) in sink
    True

    """

    def __init__(self, indicator):
        self.indicator = indicator

    def __contains__(self, segment):
        """Test whether a given segment is contained in the sink."""
        return self.indicator(segment)

    def __repr__(self):
        args = f'indicator={self.indicator!r}'
        return type(self).__name__ + '(' + args + ')'
