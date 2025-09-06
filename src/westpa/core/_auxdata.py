from collections import UserDict

import numpy as np


class AuxiliaryData(UserDict):

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('keys must be strings, not ' + type(value).__name__)
        value = np.asarray(value)
        if value.dtype == object:
            raise TypeError('object arrays are not supported')
        super().__setitem__(key, value)
