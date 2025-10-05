Propagators
===========

A *propagator* is a callable that runs dynamics for a given trajectory segment.
Its takes a :class:`~westpa.Segment` object as input, reads its ``initpoint``, sets its
``endpoint``, and returns the modified segment.

For illustration, the function below is a propagator that takes one step of a
symmetric random walk::

    import random

    def propagate(segment):
        segment.endpoint = segment.initpoint + random.choice([-1, 1])
        return segment



Built-in propagators
--------------------

.. autosummary::
   :nosignatures:

   ~westpa.OpenMMPropagator

.. autoclass:: westpa.OpenMMPropagator
.. autoclass:: westpa.OpenMMReport
