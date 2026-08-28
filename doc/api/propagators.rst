Propagators
===========

.. autosummary::
   :nosignatures:

   ~westpa.SerialPropagator
   ~westpa.VectorizedPropagator
   ~westpa.OpenMMPropagator
   ~westpa.protocols.Propagator

.. autoclass:: westpa.SerialPropagator
   :members: propagate, make_segment_dir

.. autoclass:: westpa.VectorizedPropagator
   :members: propagate, make_segment_dir

.. autoclass:: westpa.OpenMMPropagator
   :members: add_reporter

.. autoclass:: westpa.protocols.Propagator()
   :members: __call__
