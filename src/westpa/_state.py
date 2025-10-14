import numpy as np


class State:
    """Describes a specific configuration (i.e., a microstate) of the system being simulated.

    States may be created by providing a coordinate tuple, a reference to a
    file containing coordinate data, or a unique label. At least one parameter
    must be provided.

    Parameters
    ----------
    coord : 1-D array-like, optional
        Coordinate tuple.
    ref : str, optional
        Reference (e.g., an absolute path or URI) to a file containing coordinate data.
    label : str, optional
        Unique label or identifier.

    Attributes
    ----------
    coord : numpy.ndarray or None
    ref : str or None
    label : str or None

    Examples
    --------

    2-D system:

    >>> import westpa
    >>> westpa.State(coord=[0.0, 0.0])
    State(coord=[0.0, 0.0])

    Coordinates stored in a local file:

    >>> westpa.State(ref='/path/to/file.xyz')
    State(ref='/path/to/file.xyz')

    Discrete system with integer state labels:

    >>> westpa.State(label='0')
    State(label='0')

    """

    def __init__(self, *, coord=None, ref=None, label=None):
        if label is None and coord is None and ref is None:
            raise ValueError("'coord', 'ref', or 'label' must be provided")

        if coord is not None:
            coord = np.asarray(coord)
            if not coord.ndim == 1:
                raise ValueError("'coord' must be a 1-D array")

        self._coord = coord
        self._ref = str(ref) if ref is not None else None
        self._label = str(label) if label is not None else None

    @property
    def label(self):
        """State label."""
        return self._label

    @property
    def coord(self):
        """Coordinate tuple."""
        return self._coord.copy() if self._coord is not None else None

    @property
    def ref(self):
        """Reference to a file containing coordinate data."""
        return self._ref

    def __repr__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.to_dict().items())
        return type(self).__name__ + '(' + args + ')'

    def to_dict(self):
        d = {}
        for name in ('label', 'coord', 'ref'):
            if (value := getattr(self, name)) is not None:
                if name == 'coord':
                    value = value.tolist()
                d[name] = value
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
