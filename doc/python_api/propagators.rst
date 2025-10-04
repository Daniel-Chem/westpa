Propagators
===========

A *propagator* is a callable that runs dynamics for a given trajectory segment.
Its takes a :class:`~westpa.Segment` object as input, reads its ``initpoint``, sets its
``endpoint``, and returns the modified segment.

.. autoclass:: westpa.OpenMMPropagator
