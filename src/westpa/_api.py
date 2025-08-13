import copy
import io
import logging
import os
import traceback

import numpy as np

import westpa
from .core.data_manager import WESTDataManager
from .core.propagators.executable import ExecutablePropagator
from .core.sim_manager import WESimManager
from .core._rc import WESTRC  # noqa
from .work_managers import SerialWorkManager
from .work_managers.core import WorkManager

log = logging.getLogger(__name__)
rng = np.random.Generator(np.random.MT19937())


def reweight(segment, new_weight):
    """Return a copy of a segment with a new weight.

    Parameters
    ----------
    segment : Segment
        Segment to copy.
    new_weight : float
        Weight to assign the copy.

    Returns
    -------
    Segment
        Shallow copy of `segment` with weight `new_weight`.

    """
    new_weight = float(new_weight)
    if not (0 < new_weight <= 1):
        raise ValueError("'new_weight' must be positive and less than or equal to 1")
    segment = copy.copy(segment)
    segment.weight = new_weight
    return segment


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
    return [reweight(segment, new_weight) for _ in range(into)]


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
    return reweight(segment, weights.sum())


class Simulation:
    """The Simulation object provides an interface for running a WESTPA simulation.

    Parameters
    ----------
    we_driver : WEDriver
        Driver for resampling and recycling trajectory segments.
    propagator : WESTPropagator
        Propagator for running dynamics.
    work_manager : WorkManager, optional
        Work manager for launching tasks. Defaults to ``SerialWorkManager()``.
    datafile : str, default 'west.h5'
        Pathname of the HDF5 file to create for storing simulation data.
    max_total_iterations : int, optional
        Maximum number of iterations for the run.
    max_run_walltime : float, optional
        Maximum elapsed real time for the run.
    status_stream : io.TextIOBase, optional
        Stream to print status updates to. Defaults to ``stdout``.
    verbosity : {'quiet', 'verbose', 'debug', None}, optional
        Verbosity level of status output.

    """

    def __init__(
        self,
        *,
        we_driver,
        propagator,
        work_manager=None,
        datafile='west.h5',
        max_total_iterations=None,
        max_run_walltime=None,
        generate_initial_states=False,
        propagator_block_size=1,
        status_stream=None,
        verbosity=None,
    ):
        self._datafile = os.path.abspath(datafile)
        self._sim_manager = WESimManager(
            we_driver=we_driver,
            work_manager=work_manager or SerialWorkManager(),
            data_manager=WESTDataManager(system=we_driver.system, we_h5filename=self._datafile),
            gen_istates=generate_initial_states,
        )

        # workaround for wm_ops using the global rc
        westpa.rc._propagator = propagator  # noqa
        # workaround for executable.pcoord_loader() using the global rc.system
        if isinstance(propagator, ExecutablePropagator):
            westpa.rc._system = we_driver.system  # noqa

        self.max_run_walltime = max_run_walltime
        self.max_total_iterations = max_total_iterations
        self.propagator_block_size = propagator_block_size
        self.status_stream = status_stream
        self.verbosity = verbosity

    @classmethod
    def from_rcfile(cls, rcfile):
        """Construct a simulation from a run configuration file.

        Parameters
        ----------
        rcfile : str
            YAML file specifying run configuration options (e.g., 'west.cfg').

        Returns
        -------
        Simulation
            Simulation constructed using the options in `rcfile`.

        """
        rc = WESTRC()
        rc.read_config(rcfile)
        return cls(
            we_driver=rc.we_driver,
            propagator=rc.propagator,
            work_manager=rc.work_manager,
            max_total_iterations=rc.sim_manager.max_total_iterations,
            max_run_walltime=rc.sim_manager.max_run_walltime,
            status_stream=rc.status_stream,
            verbosity=rc.verbosity,
            datafile=rc.data_manager.we_h5filename,
        )

    @property
    def sim_manager(self):
        return self._sim_manager

    @property
    def _rc(self):
        return self.sim_manager.rc

    @property
    def data_manager(self):
        return self.sim_manager.data_manager

    @property
    def datafile(self):
        return self._datafile

    @datafile.setter
    def datafile(self, value):
        if self.data_manager.we_h5file is not None:
            raise ValueError(f"can't set 'datafile': already created file {self._datafile}")
        self._datafile = os.path.abspath(value)

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
    def propagator_block_size(self):
        return self.sim_manager.propagator_block_size

    @propagator_block_size.setter
    def propagator_block_size(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("'propagator_block_size' must be at least 1")
        self.sim_manager.propagator_block_size = value

    @property
    def max_run_walltime(self):
        return self.sim_manager.max_run_walltime

    @max_run_walltime.setter
    def max_run_walltime(self, value):
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("'max_run_walltime' must be positive")
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
                raise ValueError("'max_total_iterations' must be at least 1")
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
        """Initialize the simulation, taking `segments_per_state` initial states
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

        if os.path.exists(self.data_manager.we_h5filename):
            reason = f'data file {self.data_manager.we_h5filename!r} already exists'
            raise ValueError(f"can't initialize the simulation: {reason}")

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
        """Run the simulation.

        Parameters
        ----------
        n_iters : int, optional
            Number of iterations to run. If not provided, the simulation will be
            run until :attr:`max_total_iterations` or :attr:`max_run_walltime`.

        """
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
