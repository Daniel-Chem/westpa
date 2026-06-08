import logging
import subprocess
from .base import Propagator
from westpa.core.state import State
import time
import os

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_DIR_TEMPLATE = 'traj_segs/{n_iter:06d}/{seg_id:06d}'
DEFAULT_FINAL_STATE_FILENAME = 'final_state.xml'

class AmberPropagator(Propagator):
    """
    Work in progress Amber propagation engine
    """

    def __init__(
        self,
        mdin,
        prmtop,
        inpcrd,
        restart,
        mdcrd,
        amber_flags=None,
        pmemd=True,
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
            res = subprocess.run(, capture_output=True, text=True) #!need to pass the variables here for amber to not use redudnat gpu usage
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
