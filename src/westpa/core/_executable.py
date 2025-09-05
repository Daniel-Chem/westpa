import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


# This data class corresponds to the JSON type of the values in the
# ``('west', 'executable')`` section of a run configuration file.
@dataclass
class Executable:
    """An external program to be run in a subprocess.

    Parameters
    ----------
    args : str or sequence of str
    stdin : str, default os.devnull
    stdout : str, optional
    stderr : str, optional
    cwd : str, optional
    environ : Mapping[str, str], optional

    """

    args: str | Sequence[str]
    stdin: str = os.devnull
    stdout: str = None
    stderr: str = None
    cwd: str = None
    environ: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.environ = {k: str(v) for k, v in self.environ.items()}
