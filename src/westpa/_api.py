import logging
import traceback
from dataclasses import dataclass

import numpy as np

import westpa
from .core.propagators.executable import ExecutablePropagator
from .core.we_driver import WEDriver
from .core._rc import WESTRC  # noqa

log = logging.getLogger(__name__)
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
    thresholds : bool, default True
    adjust_counts : bool, default True

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
    data_manager : WESTDataManager, optional
    sim_manager : WESimManager, optional
    max_run_walltime : float, optional
    max_total_iterations : int, optional
    datafile : str, default 'west.h5'
        HDF5 file to create (or overwrite) for storage of simulation data.
    rcfile : str, optional
        YAML file specifying run configuration options.

    """

    def __init__(
        self,
        *,
        we_driver=None,
        propagator=None,
        work_manager=None,
        data_manager=None,
        sim_manager=None,
        max_run_walltime=None,
        max_total_iterations=None,
        verbosity=None,
        status_stream=None,
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

        # Workaround for wm_ops using global rc:
        westpa.rc._propagator = propagator or rc.get_propagator()  # noqa
        # Workaround for executable.pcoord_loader() using global rc:
        if isinstance(westpa.rc.propagator, ExecutablePropagator):
            westpa.rc._system = rc.get_system_driver()  # noqa

        if work_manager is not None:
            rc.work_manager = work_manager

        rc._data_manager = data_manager  # noqa
        rc.data_manager.we_h5filename = datafile

        self._sim_manager = sim_manager or rc.get_sim_manager()

        if max_run_walltime is not None:
            self.max_run_walltime = max_run_walltime
        if max_total_iterations is not None:
            self.max_total_iterations = max_total_iterations

    @property
    def sim_manager(self):
        return self._sim_manager

    @property
    def we_driver(self):
        return self.sim_manager.we_driver

    @property
    def propagator(self):
        return westpa.rc.propagator  # TODO: Decouple propagator from global rc.

    @property
    def work_manager(self):
        return self.sim_manager.work_manager

    @property
    def data_manager(self):
        return self.sim_manager.data_manager

    @property
    def max_run_walltime(self):
        return self.sim_manager.max_run_walltime

    @max_run_walltime.setter
    def max_run_walltime(self, value):
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("'max_run_walltime' must be greater than zero")
        self.sim_manager.max_run_walltime = value

    @property
    def max_total_iterations(self):
        return self.sim_manager.max_total_iterations

    @max_total_iterations.setter
    def max_total_iterations(self, value):
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("'max_total_iterations' must be an integer")
            if value <= 0:
                raise ValueError("'max_total_iterations' must be greater than zero")
        self.sim_manager.max_total_iterations = value

    # TODO: Investigate possible bug when suppress_we=True.
    def initialize(
        self,
        basis_states,
        target_states=None,
        start_states=None,
        segs_per_state=1,
        suppress_we=False,
    ):
        """Initialize the simulation, taking `segs_per_state` initial states
        from each of the given `basis_states` and `start_states`.

        Parameters
        ----------
        basis_states : list of BasisState
        target_states : list of TargetState
        start_states : list of BasisState
        segs_per_state : int, default 1
        suppress_we : bool, default False

        """
        target_states = target_states or []
        start_states = start_states or []

        # Scale basis and start state probabilities so they total to one.
        starting_states = basis_states + start_states
        scale_factor = 1 / sum(state.probability for state in starting_states)
        for state in starting_states:
            state.probability *= scale_factor

        with self.work_manager as work_manager:
            if work_manager.is_master:
                self.sim_manager.initialize_simulation(
                    basis_states=basis_states,
                    target_states=target_states,
                    start_states=start_states,
                    segs_per_state=segs_per_state,
                    suppress_we=suppress_we,
                )
            else:
                work_manager.run()

    def run(self, n_iters=None):
        with self.work_manager as work_manager:
            if work_manager.is_master:
                work_manager.install_sigint_handler()
                self.sim_manager.load_plugins()
                self.sim_manager.prepare_run()
                try:
                    self.sim_manager.run(n_iters)
                    self.sim_manager.finalize_run()
                except KeyboardInterrupt:
                    self.sim_manager.rc.pstatus('interrupted; shutting down')
                except Exception as e:
                    self.sim_manager.rc.pstatus('exception caught; shutting down')
                    if str(e) != '':
                        log.error(f'error message: {e}')
                    log.error(traceback.format_exc())
            else:
                work_manager.run()
