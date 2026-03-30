import enum
import inspect
import json
import math
from collections import UserDict

import numpy as np

from .state import State


class _AuxiliaryData(UserDict):

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('keys must be strings, not ' + type(value).__name__)
        value = np.asarray(value)
        try:
            json.dumps(value.tolist())
        except TypeError:
            raise TypeError(f'values of dtype {str(value.dtype)!r} are not supported')
        super().__setitem__(key, value)


class Segment:
    """Describes a trajectory segment.

    ``Segment`` objects should only be created directly for testing purposes.

    Attributes
    ----------
    weight : float
    initial_state : State or None
    final_state : State or None
    pcoord : numpy.ndarray or None
    data : MutableMapping[str, numpy.ndarray]
    walltime : float
    cputime : float
    n_iter : int
    seg_id : int
    parent_id : int
    wtg_parent_ids : set of int
    initpoint_type : :class:`InitPointType`
    endpoint_type : :class:`EndPointType`
    status : :class:`Status`
    failure_reason : str or None

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
        CONTINUES = 1  #: Indicates that a segment survived resampling (and recycling).
        MERGED = 2  #: Indicates that a segment was pruned during resampling.
        RECYCLED = 3  #: Indicates that a segment was recycled because it reached a sink (target).

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

    statuses = {f'SEG_STATUS_{member.name}': member.value for member in Status}
    initpoint_types = {f'SEG_INITPOINT_{member.name}': member.value for member in InitPointType}
    endpoint_types = {f'SEG_ENDPOINT_{member.name}': member.value for member in EndPointType}

    status_names = {member.value: f'SEG_STATUS_{member.name}' for member in Status}
    initpoint_type_names = {member.value: f'SEG_INITPOINT_{member.name}' for member in InitPointType}
    endpoint_type_names = {member.value: f'SEG_ENDPOINT_{member.name}' for member in EndPointType}

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
        initial_state=None,
        final_state=None,
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
        self._data = _AuxiliaryData(data or {})

        self._initial_state = initial_state
        self._final_state = final_state
        self._failure_reason = failure_reason

    @property
    def n_iter(self):
        """Iteration to which the segment belongs."""
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        if not isinstance(value, int):
            raise TypeError("'n_iter' must be an integer")
        self._n_iter = value

    @property
    def seg_id(self):
        """Segment index (0-based)."""
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
    def initial_state(self):
        """Initial state. Setting this property marks the segment as :attr:`~Segment.Status.PREPARED`."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        if not isinstance(value, State):
            raise TypeError("'initial_state' must be a State object")
        self._initial_state = value
        self.status = Segment.Status.PREPARED

    @property
    def final_state(self):
        """Final state. Setting this property marks the segment as :attr:`~Segment.Status.COMPLETE`."""
        return self._final_state

    @final_state.setter
    def final_state(self, value):
        if not isinstance(value, State):
            raise TypeError("'final_state' must be a State object")
        self._final_state = value
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
        """Progress coordinate time series.

        The value provided to ``pcoord.setter`` must be a 1-D or 2-D array.
        1-D arrays (single frames) will be converted to 2-D arrays (time
        series) of length 1.

        """
        return self._pcoord

    @pcoord.setter
    def pcoord(self, value):
        value = np.asarray(value)
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("scalar type of 'pcoord' must be numeric")
        if value.ndim == 1:
            value = value[np.newaxis, :]
        elif value.ndim != 2:
            raise ValueError("'pcoord' must be a 1-D or 2-D array")
        self._pcoord = value

    @property
    def data(self):
        """Auxiliary data."""
        return self._data

    def mark_as_failed(self, reason):
        """Mark the segment as :attr:`~Segment.Status.FAILED` due to a propagation error.

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
        return '<%s(n_iter=%r, seg_id=%r, weight=%r, parent_id=%r, wtg_parent_ids=%r) at %s>' % (
            self.__class__.__name__,
            self.n_iter,
            self.seg_id,
            self.weight,
            self.parent_id,
            self.wtg_parent_ids,
            hex(id(self)),
        )

    @property
    def initpoint_type(self):
        """Integer constant indicating the segment's origin."""
        if self.parent_id is None:
            return Segment.InitPointType.UNSET
        elif self.parent_id < 0:
            return Segment.InitPointType.NEWTRAJ
        else:
            return Segment.InitPointType.CONTINUES

    @property
    def endpoint_type(self):
        """Integer constant indicating the segment's fate."""
        return self._endpoint_type

    @endpoint_type.setter
    def endpoint_type(self, value):
        self._endpoint_type = Segment.EndPointType(value)

    @property
    def status(self):
        """Integer constant indicating the segment's propagation status."""
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
            if name == 'status' and value == Segment.Status.UNSET:
                continue

            match name:
                case 'initial_state' | 'final_state':
                    d[name] = value.to_dict()
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
                case 'initial_state' | 'final_state':
                    kwargs[name] = State.from_dict(value)
                case _:
                    kwargs[name] = value
        return cls(**kwargs)

    def __replace__(self, /, **changes):  # support copy.replace() in Python >=3.13.
        parameters = inspect.signature(self.__init__).parameters
        kwargs = {name: getattr(self, name) for name in parameters}
        kwargs.update(changes)
        return type(self)(**kwargs)

    def replace(self, **changes):
        """Return a modified copy of the segment.

        Parameters
        ----------
        changes : Mapping[str, Any]
            Attributes to modify.

        Returns
        -------
        Segment
            Copy of the segment with attributes modified according to `changes`.

        """
        return self.__replace__(**changes)

    # TODO: Remove. Use `segment.status.name` in new code.
    status_text = property((lambda s: s.status_names[s.status]))
    endpoint_type_text = property((lambda s: s.endpoint_type_names[s.endpoint_type]))
