from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """A progress coordinate or auxiliary dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    loader : {aux_data_loader, npy_data_loader, pickle_data_loader}
        Function for loading the data from disk after propagation.
    dir : bool, default False
        Whether the location on disk is a directory (True) or a file (False).
    filename : str, optional
        String with a ``{segment}`` replacement field, indicating a file
        or directory in which to store data for a given segment.
    h5path : str, optional
    dtype : numpy.dtype, optional
    scaleoffset : int, optional
    store : bool, default True
    load : bool, default False
    enabled : bool, default True
        If False, the dataset will not be loaded.

    """

    name: str
    loader: Callable
    dir: bool = False
    filename: str = None
    h5path: str = None
    dtype: np.dtype = None
    scaleoffset: int = None
    store: bool = True
    load: bool = False
    enabled: bool = True
