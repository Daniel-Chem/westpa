import json
from dataclasses import dataclass

import numpy as np

import westpa
from .core.sim_manager import WESimManager
from .core.states import BasisState, TargetState
from .core.we_driver import WEDriver
from .core._rc import WESTRC

rng = np.random.Generator(np.random.MT19937())


def split(segment, into=2):
    """Split a segment into two or more segments, distributing the weight equally.

    Parameters
    ----------
    segment : Segment
        Segment to split.
    into : int, default 2
        Number of segments to split `segment` into.

    Returns
    -------
    list of Segment
        Child segments created by splitting `segment`.

    """
    if not isinstance(into, int):
        raise TypeError("'into' must be an integer")
    if not into >= 2:
        raise ValueError("'into' must be greater than or equal to 2")
    new_weight = segment.weight / into
    return [segment.reweight(new_weight) for _ in range(into)]


def merge(segments):
    """Merge two or more segments into a single segment. The surviving segment
    is selected randomly according to weight.

    Parameters
    ----------
    segments : list of Segment
        Segments to merge.

    Returns
    -------
    Segment
        Surviving segment, assigned the total weight of `segments`.

    """
    weights = np.array([segment.weight for segment in segments])
    segment = rng.choice(segments, p=weights)
    return segment.reweight(weights.sum())


@dataclass
class ProgressCoordinate:  # TODO: Convert to regular class.
    ndim: int
    len: int
    dtype: np.dtype = np.dtype('float32')

    def __post_init__(self):
        if not isinstance(self.ndim, int):
            raise TypeError("'ndim' must be a integer")
        if self.ndim < 1:
            raise ValueError("'ndim' must be at least 1")

        if not isinstance(self.len, int):
            raise TypeError("'len' must be an integer")
        if self.len < 2:
            raise ValueError("'len' must be at least 2")

        self.dtype = np.dtype(self.dtype)


class DefaultWEDriver(WEDriver):
    """Performs resampling using a variant of the Huber & Kim algorithm [1]_.

    Parameters
    ----------
    system : WESTSystem, optional
    largest_allowed_weight : float, default 1.0
    smallest_allowed_weight : float default 1e-310
    weight_split_threshold : float, default 2.0
    weight_merge_cutoff : float, default 1.0

    References
    ----------
    .. [1] G.A. Huber, S. Kim, Biophysical Journal, Volume 70, Issue 1, 1996,
    Pages 97-110, ISSN 0006-3495, https://doi.org/10.1016/S0006-3495(96)79552-8.

    """

    def __init__(
        self,
        *,
        system=None,
        largest_allowed_weight=1.0,
        smallest_allowed_weight=1e-310,
        weight_split_threshold=2.0,
        weight_merge_cutoff=1.0,
        thresholds=True,
        adjust_counts=True,
    ):
        super().__init__(rc=WESTRC(), system=system)
        self.largest_allowed_weight = largest_allowed_weight
        self.smallest_allowed_weight = smallest_allowed_weight
        self.weight_split_threshold = weight_split_threshold
        self.weight_merge_cutoff = weight_merge_cutoff
        self.thresholds = thresholds
        self.adjust_counts = adjust_counts

    @property
    def thresholds(self):
        return self.do_thresholds

    @thresholds.setter
    def thresholds(self, value):
        if not isinstance(value, bool):
            raise TypeError("'thresholds' must be True or False")
        self.do_thresholds = value

    @property
    def adjust_counts(self):
        return self.do_adjust_counts

    @adjust_counts.setter
    def adjust_counts(self, value):
        if not isinstance(value, bool):
            raise TypeError("'adjust_counts' must be True or False")
        self.do_adjust_counts = value


class Simulation:
    """The Simulation object provides an interface for initializing and running
    a WESTPA simulation.

    Parameters
    ----------
    we_driver : WEDriver, optional
    propagator : WESTPropagator, optional
    work_manager : WorkManager, optional
    basis_states : list of tuple
        List of tuples of the form ``(identifier, probability[, auxref])``.
    start_states : list of tuple, optional
        List of tuples of the form ``(identifier, probability[, auxref])``.
    target_states : Mapping[str, ArrayLike]
        Mapping from target state labels to representative progress coordinates.
    segments_per_state : int, default 1
        Number of segments to initialize from each basis or start state.
    max_run_walltime : float, optional
    max_total_iterations : int, optional
    datafile : str, default 'west.h5'
        File for storing simulation data.
    rcfile : str, optional

    """

    def __init__(
        self,
        *,
        we_driver=None,
        propagator=None,
        work_manager=None,
        basis_states,
        start_states=None,
        target_states=None,
        segments_per_state=1,
        max_run_walltime=None,
        max_total_iterations=None,
        verbosity=None,
        status_stream=None,
        data_manager=None,
        datafile='west.h5',
        rcfile=None,
    ):
        rc = WESTRC()
        rc.verbosity = verbosity
        rc.status_stream = status_stream

        if rcfile is not None:
            rc.config.update_from_file(rcfile)

        if we_driver is not None:
            rc._we_driver = we_driver  # noqa
            rc._system = we_driver.system  # noqa

        westpa.rc._propagator = propagator or rc.get_propagator()  # noqa
        if work_manager is not None:
            rc.work_manager = work_manager

        if data_manager is not None:
            rc._data_manager = data_manager  # noqa
        rc.data_manager.we_h5filename = datafile

        sim_manager = WESimManager(rc=rc)
        if max_run_walltime is not None:
            sim_manager.max_run_walltime = max_run_walltime
        if max_total_iterations is not None:
            sim_manager.max_total_iterations = max_total_iterations

        basis_states = list(map(basis_state_from_tuple, basis_states))
        start_states = list(map(basis_state_from_tuple, start_states or []))

        if target_states is None:
            target_states = []
        else:
            target_states = list(map(target_state_from_tuple, target_states.items()))

        # Scale basis and start state probabilities so they total to one.
        starting_states = basis_states + start_states
        scale_factor = 1 / sum(state.probability for state in starting_states)
        for state in starting_states:
            state.probability *= scale_factor

        with sim_manager.work_manager as work_manager:
            if work_manager.is_master:
                sim_manager.initialize_simulation(
                    basis_states=basis_states,
                    start_states=start_states,
                    target_states=target_states,
                    segs_per_state=segments_per_state,
                    suppress_we=True,
                )
            else:
                work_manager.run()

        self._sim_manager = sim_manager


def basis_state_from_tuple(t):
    if len(t) == 2:
        identifier, probability = t
        auxref = None
    elif len(t) == 3:
        identifier, probability, auxref = t
        if not isinstance(auxref, str):
            raise TypeError("'auxref' must be a string")
    else:
        raise ValueError('tuple must be of the form (identifier, probability[, auxref])')

    try:
        label = json.dumps(identifier)
    except TypeError:
        raise TypeError("'identifier' must be JSON-serializable")
    probability = float(probability)
    if not (0 < probability <= 1):
        raise ValueError("'probability' must be positive and less than or equal to one")

    return BasisState(label, probability, auxref=auxref)


def target_state_from_tuple(t):
    label, pcoord = t
    if not isinstance(label, str):
        raise TypeError(f"'label' must be a string, not {type(label).__name__}")
    return TargetState(label, pcoord)
