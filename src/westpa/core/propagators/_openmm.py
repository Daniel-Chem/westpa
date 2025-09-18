import copy
import os

import openmm
from openmm.app import DCDReporter, Simulation, StateDataReporter

from ._abc import Propagator


class OpenMMPropagator(Propagator):
    """Molecular dynamics propagator built on the `OpenMM <https://openmm.org/>`_ toolkit.

    This propagator assumes that :class:`Microstate` identifiers are absolute
    paths to XML files containing serialized OpenMM State objects, e.g.,
    ``os.path.abspath('state.xml')``.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the system.
    system : openmm.System
        System (particles, forces, and constraints) to simulate.
    integrator : openmm.Integrator
        Integrator to use for simulating the system.
    steps : int
        Number of time steps to simulate for each segment.
    platform : openmm.Platform, optional
        Platform to use for calculations.
    platform_properties : Mapping[str, str], optional
        Platform-specific properties to pass to the simulation context.
    sim_root : str, optional
        Root directory for simulation output. Default is the current working directory.
    output_dir_template : str, default 'traj_segs/{n_iter:06d}/{seg_id:06d}'
        Template string specifying the subdirectory in which to store output
        for a given segment. Must contain ``{n_iter}``  and ``{seg_id}``
        replacement fields.
    endpoint_filename : str, default 'endpoint.xml'
        Name of the file for storing the segment's termination point.
    traj_report_interval : int, optional
        Interval (in time steps) at which to write frames to the trajectory file.
        By default, no trajectory file is written.
    traj_filename : str, default 'traj.dcd'
        Name of the trajectory file.
    traj_options : Mapping[str, Any], optional
        Keyword arguments to pass to the
        `trajectory reporter <https://docs.openmm.org/latest/api-python/generated/openmm.app.dcdreporter.DCDReporter.html>`_.
    log_report_interval : int, optional
        Interval (in time steps) at which to write state data to the log file.
        By default, no log file is written.
    log_filename : str, default 'log.csv'
        Name of the log file.
    log_options : Mapping[str, Any], optional
        Keyword arguments to pass to the `state data reporter
        <https://docs.openmm.org/latest/api-python/generated/openmm.app.statedatareporter.StateDataReporter.html>`_.

    Examples
    --------
    Create an OpenMM propagator:

    >>> import westpa
    >>> import openmm
    >>> from openmm import app, unit
    >>> topology = app.PDBFile('topology.pdb').getTopology()
    >>> forcefield = app.ForceField('amber14-all.xml')
    >>> propagator = westpa.OpenMMPropagator(
    ...     topology=topology,
    ...     system=forcefield.createSystem(
    ...         topology, nonbondedMethod=app.PME, constraints=app.HBonds
    ...     ),
    ...     integrator=openmm.LangevinIntegrator(
    ...         300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ...     ),
    ...     steps=1000,
    ... )

    Initialize a simulation with a single basis state stored in ``bstate.xml``:

    >>> simulation = westpa.Simulation(propagator=propagator, ...)
    >>> simulation.initialize(
    ...     basis_states=[westpa.Microstate(os.path.abpath('bstate.xml'))]
    ... )

    """

    def __init__(
        self,
        *,
        topology,
        system,
        integrator,
        steps,
        platform=None,
        platform_properties=None,
        sim_root=None,
        output_dir_template="traj_segs/{n_iter:06d}/{seg_id:06d}",
        endpoint_filename="endpoint.xml",
        traj_report_interval=None,
        traj_filename="traj.dcd",
        traj_options=None,
        log_report_interval=None,
        log_filename="log.csv",
        log_options=None,
    ):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.steps = steps
        self.platform = platform
        self.platform_properties = platform_properties
        self.sim_root = os.path.abspath(sim_root) if sim_root is not None else os.getcwd()
        self.output_dir_template = output_dir_template
        self.endpoint_filename = endpoint_filename
        self.traj_report_interval = traj_report_interval
        self.traj_filename = traj_filename
        self.traj_options = traj_options or {}
        self.log_report_interval = log_report_interval
        self.log_filename = log_filename
        self.log_options = log_options or {}

    def __call__(self, segment):
        simulation = Simulation(
            self.topology,
            self.system,
            copy.copy(self.integrator),
            platform=self.platform,
            platformProperties=self.platform_properties,
            state=segment.initpoint,
        )

        output_dir = os.path.join(
            self.sim_root,
            self.output_dir_template.format(n_iter=segment.n_iter, seg_id=segment.seg_id),
        )
        os.makedirs(output_dir)

        # Set up the trajectory and log reporters.
        if self.traj_report_interval is not None:
            traj_file = os.path.join(output_dir, self.traj_filename)
            simulation.reporters.append(DCDReporter(traj_file, self.traj_report_interval, **self.traj_options))
        if self.log_report_interval is not None:
            log_file = os.path.join(output_dir, self.log_filename)
            simulation.reporters.append(StateDataReporter(log_file, self.log_report_interval, **self.log_options))

        # Run the simulation and store the final state.
        try:
            simulation.step(self.steps)
        except openmm.OpenMMException as e:
            segment.mark_as_failed(f'integration error: {e}')
        else:
            endpoint_file = os.path.join(output_dir, self.endpoint_filename)
            simulation.saveState(endpoint_file)
            segment.endpoint = endpoint_file

        return segment

    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>'
