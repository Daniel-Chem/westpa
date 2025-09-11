import json
import math

import numpy as np

from ._auxdata import AuxiliaryData


class Segment:
    """A class wrapping segment data that must be passed through the work manager or data manager.
    Most fields are self-explanatory.  One item worth noting is that a negative parent ID means that
    the segment starts from the initial state with ID -(segment.parent_id+1)
    """

    SEG_STATUS_UNSET = 0
    SEG_STATUS_PREPARED = 1
    SEG_STATUS_COMPLETE = 2
    SEG_STATUS_FAILED = 3

    SEG_INITPOINT_UNSET = 0
    SEG_INITPOINT_CONTINUES = 1
    SEG_INITPOINT_NEWTRAJ = 2

    SEG_ENDPOINT_UNSET = 0
    SEG_ENDPOINT_CONTINUES = 1
    SEG_ENDPOINT_MERGED = 2
    SEG_ENDPOINT_RECYCLED = 3

    statuses = {}
    initpoint_types = {}
    endpoint_types = {}

    status_names = {}
    initpoint_type_names = {}
    endpoint_type_names = {}

    # convenience functions for binning  # TODO: Can we remove these?
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
        self.n_iter = int(n_iter) if n_iter is not None else None
        self.seg_id = int(seg_id) if seg_id is not None else None
        self.status = int(status) if status is not None else None
        self.parent_id = int(parent_id) if parent_id is not None else None
        self.endpoint_type = int(endpoint_type) if endpoint_type else self.SEG_ENDPOINT_UNSET

        self.weight = float(weight) if weight is not None else None
        self.wtg_parent_ids = set(wtg_parent_ids or ())

        self.pcoord = np.asarray(pcoord) if pcoord is not None else None
        self.walltime = walltime
        self.cputime = cputime
        self._data = data or AuxiliaryData()

        self._initpoint = initpoint
        self._endpoint = endpoint
        self._failure_reason = failure_reason

    @property
    def initpoint(self):
        """Any: Starting point of the segment."""
        return self._initpoint

    @property
    def endpoint(self):
        """Any: Ending point of the segment."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value):
        try:
            json.dumps(value)
        except TypeError:
            raise TypeError("'endpoint' must be a JSON-serializable object")
        self._endpoint = value

    @property
    def data(self):
        return self._data

    def mark_as_failed(self, reason):
        """Mark the segment as failed due to a propagator error.

        Parameters
        ----------
        reason : str
            Reason for the failure.

        """
        if not isinstance(reason, str):
            raise TypeError("'reason' must be a string")
        self.status = Segment.SEG_STATUS_FAILED
        self._failure_reason = reason

    def __repr__(self):
        return '<%s(%s) n_iter=%r seg_id=%r weight=%r parent_id=%r wtg_parent_ids=%r pcoord[0]=%r pcoord[-1]=%r>' % (
            self.__class__.__name__,
            hex(id(self)),
            self.n_iter,
            self.seg_id,
            self.weight,
            self.parent_id,
            tuple(self.wtg_parent_ids or ()),
            self.pcoord[0] if self.pcoord is not None else None,
            self.pcoord[-1] if self.pcoord is not None else None,
        )

    @property
    def initpoint_type(self):
        if self.parent_id < 0:
            return Segment.SEG_INITPOINT_NEWTRAJ
        else:
            return Segment.SEG_INITPOINT_CONTINUES

    @property
    def initial_state_id(self):
        if self.parent_id < 0:
            return -(self.parent_id + 1)
        else:
            return None

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
