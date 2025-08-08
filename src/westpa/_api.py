import io
import logging
import traceback

import numpy as np

import westpa
from .core.propagators.executable import ExecutablePropagator
from .core.yamlcfg import ConfigItemMissing
from .core._rc import WESTRC  # noqa
from .work_managers import WorkManager

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
        if rcfile is not None:
            rc.config.update_from_file(rcfile)

        if we_driver is not None:
            rc._we_driver = we_driver  # noqa
            rc._system = we_driver.system  # noqa

        if propagator is None:
            try:
                propagator = rc.get_propagator()
            except ConfigItemMissing:
                raise ValueError('a propagator must be specified')
        # Workaround for wm_ops using global rc:
        westpa.rc._propagator = propagator  # noqa
        # Workaround for executable.pcoord_loader() using global rc:
        if isinstance(westpa.rc.propagator, ExecutablePropagator):
            westpa.rc._system = rc.get_system_driver()  # noqa

        if work_manager is not None:
            rc.work_manager = work_manager

        rc._data_manager = data_manager  # noqa
        rc.data_manager.we_h5filename = datafile

        self._sim_manager = sim_manager or rc.get_sim_manager()

        self.max_run_walltime = max_run_walltime
        self.max_total_iterations = max_total_iterations
        self.verbosity = verbosity
        self.status_stream = status_stream

    @property
    def sim_manager(self):
        return self._sim_manager

    @property
    def _rc(self):
        return self.sim_manager.rc

    @property
    def data_manager(self):
        return self.sim_manager.data_manager

    def pstatus(self, *args, **kwargs):
        self._rc.pstatus(*args, **kwargs)

    @property
    def we_driver(self):
        return self.sim_manager.we_driver

    @property
    def propagator(self):
        return westpa.rc.propagator

    @property
    def work_manager(self):
        return self.sim_manager.work_manager

    @work_manager.setter
    def work_manager(self, value):
        if not isinstance(value, WorkManager):
            raise TypeError("'work_manager' must be a WorkManager object")
        self.sim_manager.work_manager = value

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

    @property
    def verbosity(self):
        return self._rc.verbosity

    @verbosity.setter
    def verbosity(self, value):
        if value not in (None, 'quiet', 'verbose', 'debug'):
            raise ValueError(f"unrecognized value for 'verbosity': {value}")
        self._rc.verbosity = value

    @property
    def status_stream(self):
        return self._rc.status_stream

    @status_stream.setter
    def status_stream(self, value):
        if value is not None:
            if not isinstance(value, io.TextIOBase):
                raise TypeError("'status_stream' must be a text stream")
        self._rc.status_stream = value

    # TODO: Investigate possible bug when suppress_we=True.
    def initialize(
        self,
        basis_states,
        target_states=None,
        start_states=None,
        segments_per_state=1,
        suppress_we=False,
    ):
        """Initialize the simulation, taking `segs_per_state` initial states
        from each of the given `basis_states` and `start_states`.

        Parameters
        ----------
        basis_states : list of BasisState
        target_states : list of TargetState
        start_states : list of BasisState
        segments_per_state : int, default 1
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
                    segs_per_state=segments_per_state,
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
                    self.pstatus('interrupted; shutting down')
                except Exception as e:
                    self.pstatus('exception caught; shutting down')
                    if str(e) != '':
                        log.error(f'error message: {e}')
                    log.error(traceback.format_exc())
            else:
                work_manager.run()
