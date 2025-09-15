import copy
import os

from openmm.app import DCDReporter, Simulation, StateDataReporter

from ._abc import Propagator


class OpenMMPropagator(Propagator):
    """Molecular dynamics propagator built on the OpenMM MD engine.

    This propagator assumes that microstate are identified by absolute paths to
    XML files containing serialized OpenMM State objects.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the molecular system.
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
        Simulation root directory. Default is the current working directory.
    output_dir_template : str, default 'traj_segs/{n_iter:06d}/{seg_id:06d}'
        Template string specifying the subdirectory in which to store output
        for a given segment. The string must contain ``{n_iter}``  and
        ``{seg_id}`` replacement fields.
    endpoint_filename : str, default 'endpoint.xml'
        Name of the file for storing the segment's termination point.
    trajectory_report_interval : int, optional
        Interval (in time steps) at which to write coordinates. If None (the default),
        no trajectory will be written.
    trajectory_filename : str, default 'traj.dcd'
        Name of the trajectory file.
    trajectory_options : Mapping[str, Any], optional
        Keyword arguments to pass to the trajectory reporter.
        See the :class:`openmm.app.DCDReporter` documentation for more information.
    log_report_interval : int, optional
        Interval (in time steps) at which to write log data. If None (the default),
        no log will be written.
    log_filename : str, default 'log.csv'
        Name of the log file.
    log_options : Mapping[str, Any], optional
        Keyword arguments to pass to the log reporter.
        See the :class:`openmm.app.StateDataReporter` documentation for more information.

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
        trajectory_report_interval=None,
        trajectory_filename="traj.dcd",
        trajectory_options=None,
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
        self.trajectory_report_interval = trajectory_report_interval
        self.trajectory_filename = trajectory_filename
        self.trajectory_options = trajectory_options or {}
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

        # The final simulation state will be saved to 'endpoint_file'.
        endpoint_file = os.path.join(output_dir, self.endpoint_filename)

        # Set up trajectory and log reporters.
        if self.trajectory_report_interval is not None:
            trajectory_file = os.path.join(output_dir, self.trajectory_filename)
            simulation.reporters.append(DCDReporter(trajectory_file, self.trajectory_report_interval, **self.trajectory_options))
        if self.log_report_interval is not None:
            log_file = os.path.join(output_dir, self.log_filename)
            simulation.reporters.append(StateDataReporter(log_file, self.log_report_interval, **self.log_options))

        # Run the simulation and store the final state.
        simulation.step(self.steps)
        simulation.saveState(endpoint_file)
        segment.endpoint = endpoint_file

        return segment

    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>'
