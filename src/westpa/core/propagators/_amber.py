import logging
import subprocess
from .subprocess import SubprocessPropagator
from ..state import State
import time
import os
import numpy as np
import shutil

# Todo: Suggest adding a propagatorerror
# from .base import Propagator, PropagatorError
# Reads as: westpa.core.propagators.base.PropagatorError: Engine None was not found.
from ..sim_manager import PropagationError

# Reads as: westpa.core.sim_manager.PropagationError: Engine None was not found.

logger = logging.getLogger(__name__)


class AmberPropagator(SubprocessPropagator):
    """Molecular dynamics propagator built for the `Amber <https://ambermd.org/>`_ molecular simulation package.

    Parameters
    ----------
    md_parameters : dict
        Dictionary of control parameters written to the Amber control input
        (``mdin``) file.

    prmtop_file : str
        Path to the Amber topology (``.prmtop``) file.

    engine : str, optional
        Amber molecular dynamics engine to use for propagation. Supported
        values are:

        - ``sander``
        - ``pmemd``
        - ``pmemd.cuda``

        Default is ``"sander"``.
    """

    DEFAULT_FINAL_STATE_FILENAME = 'restrt'

    def __init__(
        self,
        md_parameters: dict,  # dictionary of md input parameters that go in the mdin file
        prmtop_file: str,  # path to topology file
        engine: str = "sander",
        **kwargs,
    ):
        super().__init__(final_state_filename=self.DEFAULT_FINAL_STATE_FILENAME, **kwargs)
        self.md_parameters = md_parameters
        self.prmtop_file = os.path.abspath(prmtop_file)
        self.engine = engine.lower()

    def get_command(self, segment, rng):

        md_parameters = self.md_parameters.copy()  # Make a copy to avoid modifying the original dictionary

        # Check engine option is allowed
        if self.engine.lower() not in ["sander", "pmemd", "pmemd.cuda"]:
            raise PropagationError(f"Engine {self.engine} is not a valid enginge option.")
            raise PropagatorError(f"Engine {self.engine} was not found.")

        # ===== start command construction handling =====
        if not shutil.which(self.engine):
            raise PropagationError(f"Engine {self.engine} not found in PATH.")
            # raise PropagatorError(f"Engine {self.engine} was not found.")

        cmd = self.engine
        topology_flag = f"-p {self.prmtop_file}"
        input_coord_flag = f"-c {segment.initial_state.file}"
        cmd = cmd + " " + topology_flag + " " + input_coord_flag
        # ===== end command construction handling =====

        # ===== start mdin handling =====
        md_parameters['ig'] = rng.integers(2**16, dtype=np.uint16)
        if segment.initpoint_type == segment.InitPointType.NEWTRAJ:
            md_parameters['irest'] = 0
            md_parameters['ntx'] = 1
        elif segment.initpoint_type == segment.InitPointType.CONTINUES:
            md_parameters['irest'] = 1
            md_parameters['ntx'] = 5

        md_parameters_string = ',\n'.join(f'{k} = {v}' for k, v in md_parameters.items())
        title = f"Segment run n_iter={segment.n_iter}, seg_id={segment.seg_id}"
        md_parameters_string = f"{title}\n&cntrl\n{md_parameters_string}\n/\n"
        # ===== end mdin handling =====

        bash_script = f"""
printf "{md_parameters_string}" > mdin
{cmd}
"""
        # returning a bash script that will be run
        return bash_script


# Todo: Basic tests tbd:'
# utilize test ref,
# utilize dummy segment, and use the input flags, to check if outputs are produced (do in tmp directory)
