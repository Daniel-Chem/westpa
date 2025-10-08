import abc
import copy
import inspect
import logging
import math
import operator
import os
import secrets
import time
from collections.abc import Container, Iterable
from datetime import timedelta

import numpy as np

from .core.systems import WESTSystem
from .core.data_manager import WESTDataManager
from .core.sim_manager import PropagationError
from .core.segment import Segment
from .core.we_driver import ConsistencyError
from .work_managers import SerialWorkManager
from .work_managers.core import WorkManager

logger = logging.getLogger(__name__)


def split(segment, m=2):
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
    if not isinstance(m, int):
        raise TypeError("'m' must be an integer")
    if not m >= 2:
        raise ValueError("'m' must be greater than or equal to 2")
    new_weight = segment.weight / m
    return [
        Segment(
            n_iter=segment.n_iter,
            seg_id=segment.seg_id,
            weight=new_weight,
            parent_id=segment.parent_id,
            wtg_parent_ids=segment.wtg_parent_ids,
            initpoint=segment.initpoint,
            endpoint=segment.endpoint,
            pcoord=segment.pcoord,
            data=segment.data,
        )
        for _ in range(m)
    ]


def merge(segments, seed=None):
    """Merge multiple trajectory segments into a single segment. The surviving
    segment is chosen randomly according to weight.

    Parameters
    ----------
    segments : Iterable[Segment]
        Segments to merge.
    seed : int | Sequence[int] | SeedSequence | BitGenerator | Generator, optional
        Seed to initialize the random state.

    Returns
    -------
    Segment
        Copy of the surviving segment, assigned the total weight of `segments`.

    """
    segments = list(segments)
    weights = np.array([segment.weight for segment in segments])
    total_weight = weights.sum()
    rng = np.random.default_rng(seed)
    choice = rng.choice(segments, p=weights / total_weight)
    return Segment(
        n_iter=segments[0].n_iter,
        seg_id=choice.seg_id,
        weight=total_weight,
        parent_id=choice.parent_id,
        wtg_parent_ids=set.union(*(segment.wtg_parent_ids for segment in segments)),
        initpoint=choice.initpoint,
        endpoint=choice.endpoint,
        pcoord=choice.pcoord,
        data=choice.data,
    )


class Simulation:
    """Interface for initializing and running a weighted ensemble simulation.

    Parameters
    ----------
    datafile : str
        Path to the HDF5 file used to store simulation data (e.g., ``'west.h5'``).
    propagator : Callable[[Segment], Segment]
        Routine that runs dynamics for a given trajectory segment. It should
        read a segment's ``initpoint``, set its ``endpoint``, and return the
        modified segment.
    resampler : Callable[[Iterable[Segment]], Iterable[Segment]]
        Routine that takes a set of propagated trajectory segments, performs
        resampling (e.g., using the :func:`split` and :func:`merge` functions),
        and returns the resampled segments.
    pcoord_calculator : Callable[[Segment], array_like], optional
        Routine that computes the progress coordinate time series for a given
        segment. If `pcoord_calculator` is provided, it will be called for each
        segment after propagation completes, and its output will be assigned to
        the segment's  ``pcoord`` attribute prior to resampling.
    source : Source, optional
        Source (initial) distribution for source-sink boundary conditions. Must
        be provided together with `sink`.
    sink : Sink, optional
        Sink (target) state for source-sink boundary conditions. Must be provided
        together with `source`.
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
        self.data_manager = WESTDataManager()
        self.data_manager.we_h5filename = datafile
        self.data_manager.system = WESTSystem()

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
        initpoints,
        weights=None,
    ):
        """Initialize the simulation.

        Parameters
        ----------
        initpoints : Iterable[array_like | str]
            Microstates from which to start trajectories.
        weights : Iterable[float], optional
            Weights to assign the trajectories. By default, the trajectories are
            assigned equal weights.

        """
        if os.path.exists(self.datafile):
            reason = f'file {self.datafile!r} already exists'
            raise FileExistsError(f"can't initialize the simulation: {reason}")

        self.data_manager.prepare_backing()
        logger.info(f'Created HDF5 file {self.datafile!r}')

        initpoints = list(initpoints)

        if weights is None:
            weights = np.ones(len(initpoints))
        else:
            weights = np.fromiter(weights, dtype=float, count=len(initpoints))
        weights /= weights.sum()

        self.current_iter_segments = [
            Segment(
                n_iter=1,
                seg_id=index,
                weight=weight,
                parent_id=-(1 + index),
                wtg_parent_ids={-(1 + index)},
                initpoint=initpoint,
                status=Segment.Status.PREPARED,
            )
            for index, (initpoint, weight) in enumerate(zip(initpoints, weights))
        ]

        # these h5 groups are required by data_manager.prepare_iteration()
        self.data_manager.create_ibstate_group([])
        self.data_manager.save_target_states([])

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
        segment_futures = set()
        pcoord_fmap = {}  # mapping from pcoord futures to seg_ids

        for segment in segments:
            future = self.work_manager.submit(self.propagator, args=(segment,))
            futures.add(future)
            segment_futures.add(future)

        logger.info('Waiting for segments to complete...')

        while futures:
            future = self.work_manager.wait_any(futures)
            futures.remove(future)

            if future in segment_futures:
                segment = future.get_result()

                if segment.status != Segment.Status.COMPLETE:
                    logger.error(f'propagation failed for segment {segment.seg_id}')
                    raise PropagationError(f'seg_id: {segment.seg_id}, reason: {segment.failure_reason}')

                self.current_iter_segments[segment.seg_id] = segment

                if self.pcoord_calculator is not None:
                    pcoord_future = self.work_manager.submit(self.pcoord_calculator, args=(segment,))
                    pcoord_fmap[pcoord_future] = segment.seg_id
                    futures.add(pcoord_future)
                else:
                    self.data_manager.update_segments(self.n_iter, segments=[segment])

            elif future in pcoord_fmap:
                seg_id = pcoord_fmap[future]
                segment = self.current_iter_segments[seg_id]
                segment.pcoord = future.get_result()
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
                initpoint=segment.initpoint,
                endpoint=segment.endpoint,
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
            new_initpoints = self.source.random_sample(len(recycled_segments))
        else:
            recycled_segments = set()
            new_initpoints = []

        for segment in self.resampled_segments:
            parent = self.current_iter_segments[segment.seg_id]

            if segment in recycled_segments:
                parent.endpoint_type = Segment.EndPointType.RECYCLED
                parent_id = -len(new_initpoints)
                initpoint = new_initpoints.pop()
            else:
                parent.endpoint_type = Segment.EndPointType.CONTINUES
                parent_id = parent.seg_id
                initpoint = parent.endpoint

            new_segment = Segment(
                weight=segment.weight,
                initpoint=initpoint,
                parent_id=parent_id,
                wtg_parent_ids=segment.wtg_parent_ids,
                n_iter=segment.n_iter + 1,
                status=Segment.Status.PREPARED,
            )
            self.next_iter_segments.append(new_segment)

        for segment in self.current_iter_segments:
            if segment.endpoint_type == Segment.EndPointType.UNSET:
                segment.endpoint_type = Segment.EndPointType.MERGED

        self.data_manager.update_segments(self.n_iter, self.current_iter_segments)
        self.data_manager.prepare_iteration(self.n_iter + 1, self.next_iter_segments)


class Resampler(abc.ABC):
    """Abstract base class for resamplers."""

    @abc.abstractmethod
    def __call__(self, segments: Iterable[Segment]) -> Iterable[Segment]:
        """Resample a set of trajectory segments."""
        return segments  # no operation


class HuberKimResampler(Resampler):
    """Implements a modified version of the Huber-Kim algorithm. [1]_

    Parameters
    ----------
    bin_mapper : BinMapper
        Bin mapper for assigning trajectory segments to bins.
    bin_target_counts : int or sequence of int
        Number of walkers to allocate to each bin. If an integer is provided,
        the value will be applied to all the bins. If a sequence is provided,
        its length must match ``bin_mapper.nbins``.
    split_threshold : float, default 2.0
        Threshold for splitting, in units of the "ideal weight" (the total
        weight of a bin divided by the target count).
    merge_cutoff : float, default 1.0
        Cutoff for merging, in units of the "ideal weight".
    adjust_counts : bool, default True
        If True, the number of walkers in a bin will be adjusted to exactly
        match the target count, except when this would violate the `min_weight`
        or `max_weight` constraints.
    min_weight : float, default 1e-100
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.
    seed : int  | Sequence[int], optional
        Seed to initialize the random state. All integer values must be
        non-negative.

    References
    ----------
    .. [1] G.A. Huber, S. Kim, \
    Biophysical Journal, Volume 70, Issue 1, 1996, Pages 97-110, ISSN 0006-3495, https://doi.org/10.1016/S0006-3495(96)79552-8.

    """

    def __init__(
        self,
        *,
        bin_mapper,
        bin_target_counts,
        split_threshold=2.0,
        merge_cutoff=1.0,
        adjust_counts=True,
        min_weight=1e-100,
        max_weight=1.0,
        seed=None,
    ):
        self.bin_mapper = None
        self.bin_target_counts = None
        self._update_bins(bin_mapper, bin_target_counts)

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.adjust_counts = adjust_counts
        self.split_threshold = split_threshold
        self.merge_cutoff = merge_cutoff

        seed = seed if seed is not None else secrets.randbits(128)
        logger.info(f'{seed=}')
        self.rng = np.random.default_rng(seed)

    def _update_bins(self, bin_mapper, bin_target_counts):
        if isinstance(bin_target_counts, int):
            bin_target_counts = np.repeat(bin_target_counts, bin_mapper.nbins)
        else:
            bin_target_counts = np.asarray(bin_target_counts, dtype=int)
            if not len(bin_target_counts) == bin_mapper.nbins:
                raise ValueError("length of 'bin_target_counts' must equal the number of bins")

        if (bin_target_counts < 1).any():
            raise ValueError("'bin_target_counts' must be positive")

        self.bin_mapper = bin_mapper
        self.bin_target_counts = bin_target_counts

    def __call__(self, segments):
        bins_by_index = self.bin_mapper.map(segments)

        weight_getter = operator.attrgetter('weight')

        # Adapted from WEDriver._run_we():
        for index, bin_ in bins_by_index.items():
            target_count = self.bin_target_counts[index]

            segments = np.array(sorted(bin_, key=weight_getter))
            weights = np.array(list(map(weight_getter, segments)))

            ideal_weight = weights.sum() / target_count

            # Split walkers with weight > split_threshold * ideal_weight.
            to_split = weights > self.split_threshold * ideal_weight
            for segment in segments[to_split]:
                bin_.remove(segment)
                m = int(math.ceil(segment.weight / ideal_weight))
                new_segments = split(segment, m)
                bin_.update(new_segments)
            segments = segments[~to_split]
            weights = weights[~to_split]

            # Merge sets of walkers with cumulative weight <= merge_cutoff * ideal_weight.
            cumulative_weight = np.add.accumulate(weights)
            to_merge = cumulative_weight <= self.merge_cutoff * ideal_weight
            while sum(to_merge) >= 2:
                bin_.difference_update(segments[to_merge])
                new_segment = merge(segments[to_merge], seed=self.rng)
                bin_.add(new_segment)
                segments = segments[~to_merge]
                cumulative_weight = cumulative_weight[~to_merge] - new_segment.weight
                to_merge = cumulative_weight <= self.merge_cutoff * ideal_weight

            # Adjust counts. TODO: Refactor to avoid repeated sorts.
            if self.adjust_counts:
                while len(bin_) < target_count:
                    logger.debug('adjusting counts by splitting')
                    segments = sorted(bin_, key=weight_getter)
                    bin_.remove(segments[-1])
                    new_segments = split(segments[-1])  # split largest walker in 2
                    bin_.update(new_segments)
                while len(bin_) > target_count:
                    logger.debug('adjusting counts by merging')
                    segments = sorted(bin_, key=weight_getter)
                    bin_.difference_update(segments[:2])  # merge 2 smallest walkers
                    new_segment = merge(segments[:2], seed=self.rng)
                    bin_.add(new_segment)

            # Apply weight thresholds.
            segments = np.array(sorted(bin_, key=weight_getter))
            weights = np.array(list(map(weight_getter, segments)))
            to_split = weights > self.max_weight
            for segment in segments[to_split]:
                bin_.remove(segment)
                m = int(math.ceil(segment.weight / self.max_weight))
                new_segments = split(segment, m)
                bin_.update(new_segments)
            to_merge = weights < self.min_weight
            while sum(to_merge) >= 2:
                bin_.difference_update(segments[to_merge])
                new_segment = merge(segments[to_merge], seed=self.rng)
                bin_.add(new_segment)
                segments = np.array(sorted(bin_, key=weight_getter))
                weights = np.array(list(map(weight_getter, segments)))
                to_merge = weights < self.min_weight

            for segment in bin_:
                if not (self.min_weight <= segment.weight <= self.max_weight):
                    logger.warning(f'Unable to fulfill weight constraints for {segment}.')

        return set.union(*bins_by_index.values())

    def __repr__(self):
        args = []
        for name in inspect.signature(self.__init__).parameters:
            value = getattr(self, name)
            if name == 'bin_target_counts':
                if np.unique(value).size == 1:  # if counts are all the same
                    value = value[0].item()  # reduce to an integer
                else:
                    value = value.tolist()
            args.append(f'{name}={value!r}')
        return type(self).__name__ + '(' + ', '.join(args) + ')'


class Source:
    """Represents a source (initial) distribution.

    Parameters
    ----------
    microstates : Iterable[array_like | str]
        Microstates from which to initialize trajectories.
    probabilities : Iterable[float], optional
        Probability of each microstate to be selected when initializing
        a trajectory. Defaults to a uniform distribution.

    Attributes
    ----------
    microstates : Sequence[ndarray | str]
        Microstates belonging to the source.
    probabilities : NDArray[float]
        Probability of each microstate to be selected when initializing
        a trajectory.

    Methods
    -------
    random_sample

    """

    def __init__(self, microstates, probabilities=None):
        self.microstates = [x if isinstance(x, str) else np.asarray(x) for x in microstates]

        if probabilities is None:
            probabilities = np.ones(len(microstates))
        else:
            probabilities = np.fromiter(probabilities, dtype=float)
            if len(probabilities) != len(microstates):
                raise ValueError("length of 'probabilities' must match length of 'microstates'")
        probabilities /= probabilities.sum()

        self.probabilities = probabilities

    def random_sample(self, k, seed=None):
        """Generate a random sample of microstates from the source.

        Parameters
        ----------
        k : int
            Number of states to draw.
        seed : int | Sequence[int] | SeedSequence | BitGenerator | Generator, optional
            Seed to initialize the pseudo-random number generator.

        Returns
        -------
        list
            Random sample of `k` microstates.

        """
        rng = np.random.default_rng(seed)
        return rng.choice(self.microstates, p=self.probabilities, size=k).tolist()

    def __repr__(self):
        args = f'microstates={self.microstates}, probabilities={self.probabilities.tolist()}'
        return type(self).__name__ + '(' + args + ')'


class Sink(Container):
    """Represents a sink (target) state.

    Parameters
    ----------
    indicator : Callable[[Segment], bool]
        Function that returns True if a given segment terminated in the sink,
        False otherwise. This function can assume that the segment has completed
        propagation and (if using a progress coordinate) that its ``pcoord``
        attribute has been set.

    Attributes
    ----------
    indicator : Callable[[Segment], bool]
        Indicator function.

    Methods
    -------
    __contains__

    Examples
    --------
    Create a sink containing segments with a final (1-D) progress coordinate
    value greater than 1:

    >>> import westpa
    >>> sink = westpa.Sink(lambda segment: segment.pcoord[-1, 0] > 1)

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
