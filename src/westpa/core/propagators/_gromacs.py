import os

import numpy as np

from .subprocess import SubprocessPropagator


class GromacsPropagator(SubprocessPropagator):
    """Molecular dynamics propagator built on the `GROMACS <https://www.gromacs.org/>`_ package.

    Parameters
    ----------
    topology_file : str
    md_parameters : Mapping[str, Any]
    ref_structure_file : str, optional
    ref_b_structure_file : str, optional
    index_file : str, optional
    final_state_filename : str, optional
    **kwargs

    """

    DEFAULT_FINAL_STATE_FILENAME = 'confout.gro'

    def __init__(
        self,
        topology_file,
        md_parameters,
        ref_coordinate_file=None,
        ref_b_coordinate_file=None,
        index_file=None,
        final_state_filename=None,
        **kwargs,
    ):
        self.topology_file = os.path.abspath(topology_file)
        self.md_parameters = dict(md_parameters)

        self.ref_coordinate_file = os.path.abspath(ref_coordinate_file) if ref_coordinate_file else None
        self.ref_b_coordinate_file = os.path.abspath(ref_b_coordinate_file) if ref_b_coordinate_file else None
        self.index_file = os.path.abspath(index_file) if index_file else None

        final_state_filename = final_state_filename or self.DEFAULT_FINAL_STATE_FILENAME
        super().__init__(final_state_filename=final_state_filename, **kwargs)

    def get_command(self, segment, rng):
        md_parameters = self.md_parameters | {'ld-seed': rng.integers(2**16, dtype=np.uint16)}
        grompp_args = {
            'c': segment.initial_state.file,
            'p': self.topology_file,
            'r': self.ref_coordinate_file,
            'rb': self.ref_b_coordinate_file,
            'ndx': self.index_file,
        }

        md_parameters = '\n'.join(f'{k} = {v}' for k, v in md_parameters.items())
        grompp_args = ' '.join(f'-{k} {v}' for k, v in grompp_args.items() if v is not None)

        return f"""\
printf "{md_parameters}" >grompp.mdp
gmx grompp {grompp_args}
gmx mdrun -c {self.final_state_filename} -nt 1
"""
