import copy
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import openmm.app

from .base import Propagator
from westpa.core.state import State

logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_DIR_TEMPLATE = 'traj_segs/{n_iter:06d}/{seg_id:06d}'
DEFAULT_FINAL_STATE_FILENAME = 'final_state.xml'


@dataclass
class Report:
    reporter_type: type
    filename: str
    report_interval: int
    options: dict


class OpenMMPropagator(Propagator):
    """Molecular dynamics propagator built on the `OpenMM <https://openmm.org/>`_ toolkit.

    To create a :class:`~westpa.State` object compatible with this propagator,
    write an OpenMM State object to an XML file (e.g., ``state.xml``) and pass
    the absolute path to the `file` parameter::

        >>> import westpa
        >>> state = westpa.State(file='/path/to/state.xml')

    Parameters
    ----------
    topology : openmm.app.Topology
        Molecular topology (chains, residues, atoms, and bonds).
    system : openmm.System
        OpenMM System object created by applying a force field to `topology`.
    integrator : openmm.Integrator
        Integrator to use for simulating the system.
    steps : int
        Number of time steps to simulate for each segment.
    platform : openmm.Platform, optional
        Platform to use for calculations.
    platform_properties : Mapping[str, str], optional
        Platform-specific properties to pass to the simulation context.
    segment_dir_template : str, optional
        Template string specifying the directory in which to store output for a
        given segment. The string must contain ``{n_iter}`` and ``{seg_id}``
        replacement fields. If a relative path is provided, it is assumed to be
        relative to the current working directory.
    final_state_filename : str, optional
        Name of the XML file used to store the final state of a segment.
    root_seed, block_size
        See :class:`~westpa.Propagator` class documentation for details.

    Examples
    --------

    Create a propagator that runs 2 ps of Langevin dynamics at 300 K:

    >>> import westpa
    >>> import openmm.app
    >>> from openmm import unit
    >>> pdb = openmm.app.PDBFile('topology.pdb')
    >>> forcefield = openmm.app.ForceField('amber14-all.xml')
    >>> propagator = westpa.OpenMMPropagator(
    ...     topology=pdb.topology,
    ...     system=forcefield.createSystem(
    ...         pdb.topology,
    ...         nonbondedMethod=openmm.app.PME,
    ...         constraints=openmm.app.HBonds,
    ...     ),
    ...     integrator=openmm.LangevinIntegrator(
    ...         temperature=300 * unit.kelvin,
    ...         frictionCoeff=1 / unit.picosecond,
    ...         stepSize=2 * unit.femtosecond,
    ...     ),
    ...     steps=1000,
    ... )

    Add a reporter that writes the positions of the first 22 atoms every 100 steps:

    >>> propagator.add_reporter(
    ...     openmm.app.XTCReporter,
    ...     filename='traj.xtc',
    ...     report_interval=100,
    ...     options={'atomSubset': list(range(22))},
    ... )

    Add a reporter that writes the kinetic and potential energy every 500 steps:

    >>> propagator.add_reporter(
    ...     openmm.app.StateDataReporter,
    ...     filename='log.csv',
    ...     report_interval=500,
    ...     options={'kineticEnergy': True, 'potentialEnergy': True},
    ... )

    Notes
    -----

    To use this propagator, the `OpenMM <https://pypi.org/project/OpenMM/>`_
    package must be installed, e.g.:

    .. code-block:: shell

       conda install conda-forge::openmm

    or:

    .. code-block:: shell

       pip install openmm

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
        segment_dir_template=DEFAULT_SEGMENT_DIR_TEMPLATE,
        final_state_filename=DEFAULT_FINAL_STATE_FILENAME,
        root_seed=None,
        block_size=None,
    ):
        super().__init__(root_seed=root_seed, block_size=block_size)
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.steps = steps
        self.platform = platform
        self.platform_properties = platform_properties
        self.segment_dir_template = os.path.abspath(segment_dir_template)
        self.final_state_filename = final_state_filename
        self._reports = []

    def add_reporter(self, reporter_type, filename, report_interval, options=None):
        """Add a reporter to output per-segment data.

        Parameters
        ----------
        reporter_type : type
            Class compatible with the OpenMM Reporter protocol. It must be possible
            to create an instance by passing ``filename, report_interval, **options``
            to the `reporter_type` constructor.
        filename : str
            Name of the file to write output to.
        report_interval : int
            Interval (in time steps) at which to write frames.
        options : dict[str, Any], optional
            Keyword arguments to pass to the `reporter_type` constructor.

        """
        report = Report(reporter_type, filename, report_interval, options or {})
        self._reports.append(report)

    def propagate(self, segment):
        start_time = time.time()

        integrator = copy.copy(self.integrator)  # copy to avoid 'already bound to Context' error

        rng = np.random.default_rng(self.get_child_seed(segment))
        if hasattr(integrator, 'setRandomNumberSeed'):
            integrator.setRandomNumberSeed(rng.integers(low=1, high=2**31))
        for force in self.system.getForces():
            if hasattr(force, 'setRandomNumberSeed'):
                force.setRandomNumberSeed(rng.integers(low=1, high=2**31))

        simulation = openmm.app.Simulation(
            self.topology,
            self.system,
            integrator,
            platform=self.platform,
            platformProperties=self.platform_properties,
            state=segment.initial_state.file,
        )

        segment_dir = self.segment_dir_template.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        os.makedirs(segment_dir)

        for report in self._reports:
            file = os.path.join(segment_dir, report.filename)
            reporter = report.reporter_type(file, report.report_interval, **report.options)
            simulation.reporters.append(reporter)

        try:
            simulation.step(self.steps)
        except openmm.OpenMMException as e:
            segment.mark_as_failed(str(e))
        else:
            final_state_file = os.path.join(segment_dir, self.final_state_filename)
            simulation.saveState(final_state_file)
            segment.final_state = State(file=final_state_file)
            segment.walltime = time.time() - start_time

        return segment
