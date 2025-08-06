import os

import openmm.app

from ._abc import Propagator


class OpenMMPropagator(Propagator):
    """Molecular dynamics propagator built on the OpenMM MD engine.

    This propagator assumes that segment ``initpoint`` and ``endpoint`` values
    are absolute paths to files containing XML-serialized OpenMM State objects.

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
    output_dir_template : str, default 'traj_segs/{segment.n_iter:06d}/{segment.seg_id:06d}'
        Directory in which to store output for a given segment.
    endpoint_filename : str, default 'endpoint.xml'
        Name of the file for storing the segment's termination point.
    trajectory_report_interval : int, optional
        Interval (in time steps) at which to write coordinates. If None (the default),
        no trajectory will be written.
    trajectory_filename : str, default 'seg.dcd'
        Name of the trajectory file.
    trajectory_options : Mapping[str, Any], optional
        Keyword arguments to pass to the trajectory reporter.
        See the :class:`openmm.app.DCDReporter` documentation for more information.
    log_report_interval : int, optional
        Interval (in time steps) at which to write log data. If None (the default),
        no log will be written.
    log_filename : str, default 'seg.log'
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
        platform=None,
        platform_properties=None,
        steps,
        sim_root=None,
        output_dir_template="traj_segs/{segment.n_iter:06d}/{segment.seg_id:06d}",
        endpoint_filename="endpoint.xml",
        trajectory_report_interval=None,
        trajectory_filename="seg.dcd",
        trajectory_options=None,
        log_report_interval=None,
        log_filename="seg.log",
        log_options=None,
    ):
        self.sim_root = os.path.abspath(sim_root) if sim_root is not None else os.getcwd()
        self.simulation = openmm.app.Simulation(
            topology,
            system,
            integrator,
            platform=platform,
            platformProperties=platform_properties,
        )
        self.steps = steps
        self.output_dir_template = output_dir_template
        self.endpoint_filename = endpoint_filename
        self.trajectory_report_interval = trajectory_report_interval
        self.trajectory_filename = trajectory_filename
        self.trajectory_options = trajectory_options or {}
        self.log_report_interval = log_report_interval
        self.log_filename = log_filename
        self.log_options = log_options or {}

    def __call__(self, segment):
        output_dir = os.path.join(self.sim_root, self.output_dir_template.format(segment=segment))
        os.makedirs(output_dir)

        # The final simulation state will be saved to 'endpoint_file'.
        endpoint_file = os.path.join(output_dir, self.endpoint_filename)

        # Set up trajectory and log reporters.
        self.simulation.reporters.clear()
        trajectory_file = None
        log_file = None
        if self.trajectory_report_interval is not None:
            trajectory_file = os.path.join(output_dir, self.trajectory_filename)
            self.simulation.reporters.append(
                openmm.app.DCDReporter(trajectory_file, self.trajectory_report_interval, **self.trajectory_options)
            )
        if self.log_report_interval is not None:
            log_file = os.path.join(output_dir, self.log_filename)
            self.simulation.reporters.append(openmm.app.StateDataReporter(log_file, self.log_report_interval, **self.log_options))

        # Run the simulation.
        self.simulation.loadState(segment.initpoint)
        self.simulation.step(self.steps)
        self.simulation.saveState(endpoint_file)

        # Store the results.
        segment.endpoint = os.path.abspath(endpoint_file)
        if trajectory_file is not None:
            segment.data["trajectory"] = os.path.abspath(trajectory_file)
        if log_file is not None:
            segment.data["log"] = os.path.abspath(log_file)

        return segment
