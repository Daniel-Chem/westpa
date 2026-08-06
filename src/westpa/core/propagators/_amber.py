import logging
import subprocess
from .base import Propagator
from .subprocess import SubprocessPropagator
from ..state import State
import time
import os
import numpy as np
import shutil

logger = logging.getLogger(__name__)

#Operate under the assumption that environment where subprocess is run has the cuda visible devices set

#Basic tests tbd:'
# utilize test ref, 
#utilize dummy segment, and use the input flags, to check if outputs are produced (do in tmp directory)


class AmberPropagator(SubprocessPropagator):
    DEFAULT_FINAL_STATE_FILENAME = 'restrt'
    DEFAULT_TRAJECTORY_FILENAME = 'mdcrd'
    def __init__(
            self,

            md_parameters:dict, #dictionary of md input parameters that go in the mdin file
            prmtop_file:str, #path to topology file
            engine:str = "sander", 
            **kwargs
    ):
        super().__init__(final_state_filename=self.DEFAULT_FINAL_STATE_FILENAME, **kwargs)
        self.md_parameters = md_parameters
        self.prmtop_file = os.path.abspath(prmtop_file)
        self.engine = engine

    def get_command(self, segment, rng):
        
        md_parameters=self.md_parameters.copy()  # Make a copy to avoid modifying the original dictionary

        # ===== start command construction handeling =====
        engine: str | None=shutil.which(self.engine)
        if engine not in ["sander","pmemd"]:
            raise RuntimeError(f"Engine {self.engine} was not found.")
        cmd=self.engine
        topology_flag= f"-p {self.prmtop_file}"
        input_coord_flag=f"-c {segment.initial_state.file}"
        cmd=(cmd+" "+topology_flag+" "+input_coord_flag)
        # ===== end command construction handeling =====


        # ===== start mdin handeling =====
        if segment.initpoint_type == segment.InitPointType.NEWTRAJ:
            md_parameters['irest'] = 0
            md_parameters['ntx'] = 1
            md_parameters['ig'] = rng.integers(2**16, dtype=np.uint16)
        elif segment.initpoint_type == segment.InitPointType.CONTINUES:
            md_parameters['irest'] = 1
            md_parameters['ntx'] = 5
            md_parameters['ig'] = rng.integers(2**16, dtype=np.uint16)
        
        md_parameters_string = ',\n'.join(f'{k} = {v}' for k, v in md_parameters.items())
        title=f"Segment run n_iter={segment.n_iter}, seg_id={segment.seg_id}"
        md_parameters_string = f"{title}\n&cntrl\n{md_parameters_string}\n/\n"
        # ===== end mdin handeling =====

        bash_script=f"""
printf {cmd} > cmd.txt
printf "{md_parameters_string}" > mdin
{cmd} 
"""
        #returning a bash script that will be run
        return bash_script
