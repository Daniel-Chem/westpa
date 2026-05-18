import logging
import subprocess 
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

class Amberropagator(Propagator):
    """
    Work in progress Amber propagation engine 

    """

    def __init__(
        self,
        mdin,
        inpcrd,
        mdcrd,
        mdout=None,
        mdinfo=None,
        prmtop=None,
        refc=None,
        mtmd=None,
        mdcrt=None,
        inptraj=None,
        mdvel=None,
        mdfrc=None,
        mden=None,
        restrt=None,
        inpdip=None,
        rstdip=None,
        cpin=None,
        cpestrt=None,
        cpout=None,
        cein=None,
        ceresrst=None,
        cecout=None,
        evbin=None,
        suffix=None,
        segment_dir_template=None,
        final_state_filename=None,
        root_seed=None,
        **kwargs
    ):
        super().__init__()
        self.mdin = mdin
        self.mdout = mdout
        self.mdinfo = mdinfo
        self.prmtop = prmtop
        self.inpcrd = inpcrd
        self.refc = refc
        self.mtmd = mtmd
        self.mdcrt = mdcrt
        self.inptraj = inptraj
        self.mdvel = mdvel
        self.mdfrc = mdfrc
        self.mden = mden
        self.restrt = restrt
        self.inpdip = inpdip
        self.rstdip = rstdip
        self.cpin = cpin
        self.cpestrt = cpestrt
        self.cpout = cpout
        self.cein = cein
        self.ceresrst = ceresrst
        self.cecout = cecout
        self.evbin = evbin
        self.suffix = suffix
        #!Figure out if **kwargs

        #self.platform_properties = platform_properties
        #self.segment_dir_template = os.path.abspath(segment_dir_template or DEFAULT_SEGMENT_DIR_TEMPLATE)
        #self.final_state_filename = final_state_filename or DEFAULT_FINAL_STATE_FILENAME
        #self._reports = []


    def propagate(self,segment):
        start_time = time.time()
        #Command construction
        input_flags=vars(self)
        cmd=["pmemd.cuda"]
        for k, v in input_flags.items():
            
            if v is None:
                #print(k) #all the ones that are none
                continue

            if k == "_root_seed":
                continue
            if k == "_block_size":
                continue
            
            cmd.append(f"--{k} {v}")
        #print(cmd)

        #command execution
        res = subprocess.run(cmd, capture_output=True, text=True)

        state=segment.initial_state.file #file in storage that md will pick up from #!This is specified with input flag, c 
        segment_dir = self.segment_dir_template.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        os.makedirs(segment_dir)

        final_state_file = os.path.join(segment_dir, self.mdcrd) #aka seg.nc 
        segment.final_state = State(file=final_state_file)
        segment.walltime = time.time() - start_time #done

        return segment







#$PMEMD -O -p parent.prmtop -i prod.in -c parent.rst -o seg.out -inf seg.nfo -l seg.log -x seg.nc -r seg.rst || exit 1

