Trajectory data
===============

.. autosummary::
   :nosignatures:

   ~westpa.State
   ~westpa.Segment

.. autoclass:: westpa.State

.. autoclass:: westpa.Segment
   :members: mark_as_failed, initial_state, final_state, pcoord

.. autoclass:: westpa.Segment.InitPointType()
   :noindex:
   :show-inheritance:
   :members: UNSET, CONTINUES, NEWTRAJ

.. autoclass:: westpa.Segment.EndPointType()
   :noindex:
   :show-inheritance:
   :members: UNSET, CONTINUES, MERGED, RECYCLED

.. autoclass:: westpa.Segment.Status()
   :noindex:
   :show-inheritance:
   :members: UNSET, PREPARED, COMPLETE, FAILED
