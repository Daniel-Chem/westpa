import enum
import inspect
import json
import math

import numpy as np

from ._auxdata import AuxiliaryData


class Segment:
    """Describes a trajectory segment.

    :class:`Segment` objects should only be created directly for testing purposes.

    Attributes
    ----------
    weight : float, optional
    initpoint : `JSON encodable <https://docs.python.org/3/library/json.html#py-to-json-table>`_ object
    endpoint : `JSON encodable <https://docs.python.org/3/library/json.html#py-to-json-table>`_ object
    pcoord : ndarray, optional
    data : MutableMapping[str, ndarray]
    walltime : float
    cputime : float
    n_iter : int, optional
    seg_id : int, optional
    parent_id : int, optional
    wtg_parent_ids : set of int
    initpoint_type : :class:`InitPointType`
    endpoint_type : :class:`EndPointType`
    status : :class:`Status`
    failure_reason : str, optional

    """

    class Status(enum.IntEnum):
        """Integer code indicating a segment's propagation status."""

        UNSET = 0  #: Null value.
        PREPARED = 1  #: Indicates that a segment is prepared for propagation.
        COMPLETE = 2  #: Indicates that propagation completed successfully.
        FAILED = 3  #: Indicates that propagation failed.

    class InitPointType(enum.IntEnum):
        """Integer code indicating a segment's origin."""

        UNSET = 0  #: Null value.
        CONTINUES = 1  #: Indicates that a segment continues an existing trajectory.
        NEWTRAJ = 2  #: Indicates that a segment begins a new trajectory.

    class EndPointType(enum.IntEnum):
        """Integer code indicating a segment's fate."""

        UNSET = 0  #: Null value.
        CONTINUES = 1  #: Indicates that a segment survived resampling.
        MERGED = 2  #: Indicates that a segment was pruned during resampling.
        RECYCLED = 3  #: Indicates that a segment was recycled because it reached the sink (target) state.

    SEG_STATUS_UNSET = Status.UNSET
    SEG_STATUS_PREPARED = Status.PREPARED
    SEG_STATUS_COMPLETE = Status.COMPLETE
    SEG_STATUS_FAILED = Status.FAILED

    SEG_INITPOINT_UNSET = InitPointType.UNSET
    SEG_INITPOINT_CONTINUES = InitPointType.CONTINUES
    SEG_INITPOINT_NEWTRAJ = InitPointType.NEWTRAJ

    SEG_ENDPOINT_UNSET = EndPointType.UNSET
    SEG_ENDPOINT_CONTINUES = EndPointType.CONTINUES
    SEG_ENDPOINT_MERGED = EndPointType.MERGED
    SEG_ENDPOINT_RECYCLED = EndPointType.RECYCLED

    statuses = {}
    initpoint_types = {}
    endpoint_types = {}

    status_names = {}
    initpoint_type_names = {}
    endpoint_type_names = {}

    # convenience functions for binning  # TODO: Remove.
    @staticmethod
    def initial_pcoord(segment):
        """Return the initial progress coordinate point of this segment."""
        return segment.pcoord[0]

    @staticmethod
    def final_pcoord(segment):
        """Return the final progress coordinate point of this segment."""
        return segment.pcoord[-1]

    def __init__(
        self,
        n_iter=None,
        seg_id=None,
        weight=None,
        endpoint_type=None,
        parent_id=None,
        wtg_parent_ids=None,
        pcoord=None,
        status=None,
        walltime=None,
        cputime=None,
        data=None,
        initpoint=None,
        endpoint=None,
        failure_reason=None,
    ):
        # NaNs appear sometimes if a WEST program is terminated unexpectedly; replace with zero
        walltime = 0.0 if walltime is None or math.isnan(walltime) else walltime
        cputime = 0.0 if cputime is None or math.isnan(cputime) else cputime

        # the int() and float() calls are required so that new-style string formatting doesn't barf
        # assuming that the respective fields are actually strings, probably after implicitly
        # calling __str__() on them.  Not sure if this is a numpy, h5py, or python problem
        self._n_iter = int(n_iter) if n_iter is not None else None
        self._seg_id = int(seg_id) if seg_id is not None else None
        self._status = Segment.Status(status) if status else Segment.Status.UNSET
        self._parent_id = int(parent_id) if parent_id is not None else None
        self._endpoint_type = Segment.EndPointType(endpoint_type) if endpoint_type else Segment.EndPointType.UNSET

        self._weight = float(weight) if weight is not None else None
        self._wtg_parent_ids = set(wtg_parent_ids or ())

        self._pcoord = np.asarray(pcoord) if pcoord is not None else None
        self._walltime = walltime
        self._cputime = cputime
        self._data = AuxiliaryData(data or {})

        self._initpoint = initpoint
        self._endpoint = endpoint
        self._failure_reason = failure_reason

    @property
    def n_iter(self):
        """Iteration in which the segment was created."""
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        if not isinstance(value, int):
            raise TypeError("'n_iter' must be an integer")
        self._n_iter = value

    @property
    def seg_id(self):
        """Index of the segment (0-based)."""
        return self._seg_id

    @seg_id.setter
    def seg_id(self, value):
        if not isinstance(value, int):
            raise TypeError("'seg_id' must be an integer")
        if value < 0:
            raise TypeError("'seg_id' must be non-negative")
        self._seg_id = value

    @property
    def parent_id(self):
        """Index of the segment's parent."""
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value):
        if not isinstance(value, int):
            raise TypeError("'parent_id' must be an integer")
        self._parent_id = value

    @property
    def wtg_parent_ids(self):
        """Indices of the walkers that contributed weight to the segment."""
        return self._wtg_parent_ids

    @wtg_parent_ids.setter
    def wtg_parent_ids(self, value):
        value = set(value)
        if not all(isinstance(item, int) for item in value):
            raise TypeError("items in 'wtg_parent_ids' must be integers")
        self._wtg_parent_ids = value

    @property
    def initpoint(self):
        """Initial microstate."""
        return self._initpoint

    @property
    def endpoint(self):
        """Final microstate (set by the propagator). Setting ``endpoint`` marks the segment as complete."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value):
        try:
            json.dumps(value)
        except TypeError:
            raise TypeError("'endpoint' must be a JSON encodable object")
        self._endpoint = value
        self.status = Segment.Status.COMPLETE

    @property
    def weight(self):
        """Statistical weight."""
        return self._weight

    @weight.setter
    def weight(self, value):
        value = float(value)
        if not (0 <= value <= 1):
            raise ValueError("'weight' must be between 0 and 1")
        self._weight = value

    @property
    def pcoord(self):
        """Progress coordinate time series."""
        return self._pcoord

    @pcoord.setter
    def pcoord(self, value):
        if value is not None:
            value = np.asarray(value)
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError("scalar type of 'pcoord' must be numeric")
            if value.ndim != 2:
                raise ValueError("'pcoord' must be a 2-D array")
        self._pcoord = value

    @property
    def data(self):
        """Auxiliary data."""
        return self._data

    def mark_as_failed(self, reason):
        """Mark the segment as failed due to a propagation error.

        Parameters
        ----------
        reason : str
            Reason for the failure.

        """
        if not isinstance(reason, str):
            raise TypeError("'reason' must be a string")
        self.status = Segment.Status.FAILED
        self._failure_reason = reason

    @property
    def failure_reason(self):
        """Reason (if any) why propagation failed."""
        return self._failure_reason

    @property
    def walltime(self):
        """Wall-clock time taken for propagation (zero by default)."""
        return self._walltime

    @walltime.setter
    def walltime(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("'walltime' must be positive")
        self._walltime = value

    @property
    def cputime(self):
        """Process time taken for propagation (zero by default)."""
        return self._cputime

    @cputime.setter
    def cputime(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("'cputime' must be positive")
        self._cputime = value

    def __repr__(self):
        args = ', '.join(f'{name}={value!r}' for name, value in self.to_dict().items())
        return type(self).__name__ + '(' + args + ')'

    @property
    def initpoint_type(self):
        """Whether the segment begins a new trajectory or continues an existing one."""
        if self.parent_id is None:
            return Segment.InitPointType.UNSET
        elif self.parent_id < 0:
            return Segment.InitPointType.NEWTRAJ
        else:
            return Segment.InitPointType.CONTINUES

    @property
    def endpoint_type(self):
        """Whether the segment survived to the next iteration, was merged away during
        resampling, or was recycled because it reached the sink (target) state."""
        return self._endpoint_type

    @endpoint_type.setter
    def endpoint_type(self, value):
        self._endpoint_type = Segment.EndPointType(value)

    @property
    def status(self):
        """Propagation status."""
        return self._status

    @status.setter
    def status(self, value):
        self._status = Segment.Status(value)

    @property
    def initial_state_id(self):
        if self.parent_id < 0:
            return -(self.parent_id + 1)
        else:
            return None

    def to_dict(self):
        """Serialize the segment to a JSON encodable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the segment.

        See Also
        --------
        from_dict

        """
        d = {}
        for name in inspect.signature(self.__init__).parameters:
            if (value := getattr(self, name)) is None:
                continue
            if name in ('walltime', 'cputime') and value == 0.0:
                continue
            if name in ('data', 'wtg_parent_ids') and len(value) == 0:
                continue
            if name == 'endpoint_type' and value == Segment.EndPointType.UNSET:
                continue
            match name:
                case 'pcoord':
                    d[name] = value.tolist()
                case 'data':
                    d[name] = {k: v.tolist() for k, v in value.items()}
                case 'wtg_parent_ids':
                    d[name] = list(value)
                case _:
                    d[name] = value
        return d

    @classmethod
    def from_dict(cls, d):
        """Deserialize a segment from its dictionary representation.

        Returns
        -------
        Segment
            Deserialized segment.

        See Also
        --------
        to_dict

        """
        kwargs = {}
        for name, value in d.items():
            match name:
                case 'pcoord':
                    kwargs[name] = np.asarray(value)
                case 'data':
                    kwargs[name] = {k: np.asarray(v) for k, v in value.keys()}
                case 'wtg_parent_ids':
                    kwargs[name] = set(value)
                case _:
                    kwargs[name] = value
        return cls(**kwargs)

    status_text = property((lambda s: s.status_names[s.status]))
    endpoint_type_text = property((lambda s: s.endpoint_type_names[s.endpoint_type]))


Segment.statuses.update({_attr: getattr(Segment, _attr) for _attr in dir(Segment) if _attr.startswith('SEG_STATUS_')})
Segment.initpoint_types.update({_attr: getattr(Segment, _attr) for _attr in dir(Segment) if _attr.startswith('SEG_INITPOINT_')})
Segment.endpoint_types.update({_attr: getattr(Segment, _attr) for _attr in dir(Segment) if _attr.startswith('SEG_ENDPOINT_')})

Segment.status_names.update({getattr(Segment, _attr): _attr for _attr in dir(Segment) if _attr.startswith('SEG_STATUS_')})
Segment.initpoint_type_names.update(
    {getattr(Segment, _attr): _attr for _attr in dir(Segment) if _attr.startswith('SEG_INITPOINT_')}
)
Segment.endpoint_type_names.update({getattr(Segment, _attr): _attr for _attr in dir(Segment) if _attr.startswith('SEG_ENDPOINT_')})
