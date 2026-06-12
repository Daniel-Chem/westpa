__all__ = [
    'State',
    'Segment',
    'Resampler',
    'MultinomialResampler',
    'ResidualResampler',
    'HuberKimResampler',
    'Bin',
    'BinMapper',
    'FuncBinMapper',
    'MABBinMapper',
    'PiecewiseBinMapper',
    'RectilinearBinMapper',
    'RecursiveBinMapper',
    'VectorizingFuncBinMapper',
    'VoronoiBinMapper',
    'AdaptiveVoronoiBinMapper',
    'Simulation',
    'Source',
    'Sink',
    'Propagator',
    'GromacsPropagator',
    'OpenMMPropagator',
    'Amberropagator' 'Plugin',
    'TrajectoryTree',
    'Trajectory',
    'WESTSystem',
    'BasisState',
    'TargetState',
    '_rc',
]

import shutil

from .core.state import State
from .core.segment import Segment
from .core.propagators import Propagator
from .core.binning import (
    Bin,
    BinMapper,
    FuncBinMapper,
    PiecewiseBinMapper,
    RectilinearBinMapper,
    RecursiveBinMapper,
    VectorizingFuncBinMapper,
    VoronoiBinMapper,
    MABBinMapper,
    AdaptiveVoronoiBinMapper,
)
from .core.resamplers import (
    Resampler,
    HuberKimResampler,
    MultinomialResampler,
    ResidualResampler,
)
from .core.simulation import Simulation
from .core.source_sink import Source, Sink
from .core.plugins import Plugin

if shutil.which('gmx'):
    from .core.propagators._gromacs import GromacsPropagator

try:
    from .core.propagators._openmm import OpenMMPropagator
except ImportError:
    pass

try:
    from .core.propagators._amber import AmberPropagator
except ImportError:
    pass


from .analysis import TrajectoryTree, Trajectory

from .core.states import BasisState, TargetState
from .core.systems import WESTSystem
from .core import _rc

from ._version import get_versions

rc = _rc.WESTRC()

__version__ = get_versions()["version"]

del get_versions
