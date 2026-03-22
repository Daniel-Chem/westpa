Segments
========

WESTPA's propagation and resampling routines operate on :class:`~westpa.Segment`
objects, which describe trajectory segments in various stages of completion.

.. autoclass:: westpa.Segment
   :members: initial_state, final_state, pcoord, mark_as_failed

.. autoclass:: westpa.Segment.InitPointType()
   :show-inheritance:
   :members: UNSET, CONTINUES, NEWTRAJ

.. autoclass:: westpa.Segment.EndPointType()
   :show-inheritance:
   :members: UNSET, CONTINUES, MERGED, RECYCLED

.. autoclass:: westpa.Segment.Status()
   :show-inheritance:
   :members: UNSET, PREPARED, COMPLETE, FAILED
