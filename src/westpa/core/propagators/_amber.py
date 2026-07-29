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

#Operate under the assumption that enviroment where subprocess is run has the cudavisiible devices set

#Basic tests tbd:
# utilize test ref, 
#utilize dummy segment, and use the input flags, to check if outputs are produced (do in tmp directory)


class AmberPropagator(SubprocessPropagator):
    DEFAULT_FINAL_STATE_FILENAME = 'restrt'
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
        engine=shutil.which(self.engine)
        cmd=[engine]
        topology_flag= f"-p {self.prmtop_file}"
        cmd=cmd.append(topology_flag)
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
        md_parameters_string = f"{title}\n&cntrl\n{md_parameters_string}\n/"
        # ===== end mdin handeling =====

        #returning a bash script that will be run
        return f"""
printf "{md_parameters_string}" > mdin
{cmd} 
        """












#TODO: check if there are other propagator engine, 
class AmberPropagator(Propagator):
    """
    Work in progress Amber propagation engine
    """

    def __init__( #constructor (in name)
        self,
        mdin,
        prmtop, #!combine an determine if inpcrd is needed to construct propagator
        inpcrd,
        restart,
        mdcrd, #! Combine, final state file flag instead, then determine what type it is
        amber_flags=None,
        pmemd=True, #!relace with engine option that defaults, engine will be string, can be replaced by user
        sander=False,
        segment_dir_template=DEFAULT_SEGMENT_DIR_TEMPLATE,
        final_state_filename=DEFAULT_FINAL_STATE_FILENAME,
        **kwargs
    ):
        super().__init__()
        self.mdin =mdin
        self.prmtop = prmtop
        self.restart = restart
        self.mdcrd = mdcrd
        self.amber_flags = amber_flags

        self.pmemd = pmemd
        self.sander = sander
        self.segment_dir_template = segment_dir_template
        self.final_state_filename = final_state_filename
        self.validate_inputs()

    def validate_inputs(self):
        if self.pmemd and self.sander:
            raise ValueError("Cannot specify both pmemd and sander as True. Please choose one.")
        if not self.pmemd and not self.sander:
            raise ValueError("Must specify either pmemd or sander as True. Please choose one.")

    def construct_command(self): #TODO: Need to pass the input flags explicitly not just the dictionary
        #!segment check the init point type NEWTRAJ (possibly one flag for both)
        #! check irest flag

        #!change such that the user specifies the engine
        if self.pmemd:
            cmd=["pmemd.cuda"]
        elif self.sander:
            cmd=["sander"]

        if self.amber_flags:
            for flag_key, flag_value in self.amber_flags.items():
                if flag_value is None:
                    continue
                if len(flag_key) == 1:
                    cmd.append(f"-{flag_key} {flag_value}")
                else:
                    cmd.append(f"--{flag_key} {flag_value}")
        
        return cmd

    def propagate(self,segment): #!would need to list all gpus and work manager index so taht there is no overlap within runseg
        start_time = time.time()

        cmd=self.construct_command()

        #print(f"Executing command: {cmd} ")

        #Execute the command and capture output, handling exceptions
        try:
            res = subprocess.run(cmd, capture_output=True, text=True) #!need to pass the variables here for amber to not use redudnat gpu usage
        except Exception as e:
            segment.mark_as_failed(str(e))
            return segment


        segment_dir = self.segment_dir_template.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        os.makedirs(segment_dir)
        final_state_file = os.path.join(segment_dir,self.restart)
        segment.final_state = State(file=final_state_file)
        segment.walltime = time.time() - start_time #done
        return segment







#$PMEMD -O -p parent.prmtop -i prod.in -c parent.rst -o seg.out -inf seg.nfo -l seg.log -x seg.nc -r seg.rst || exit 1
