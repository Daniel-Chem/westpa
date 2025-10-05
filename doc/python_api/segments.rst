Segment objects
===============

Propagation, resampling, and other WESTPA simulation routines operate on
:class:`~westpa.Segment` objects, which describe trajectory segments in various
stage of completion.

.. autoclass:: westpa.Segment
   :members: endpoint, mark_as_failed

.. autoclass:: westpa.Segment.InitPointType()
   :show-inheritance:
   :members: CONTINUES, NEWTRAJ

.. autoclass:: westpa.Segment.EndPointType()
   :show-inheritance:
   :members: UNSET, CONTINUES, MERGED, RECYCLED

.. autoclass:: westpa.Segment.Status()
   :show-inheritance:
   :members: UNSET, PREPARED, COMPLETE, FAILED
