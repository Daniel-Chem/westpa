Propagators
===========

A *propagator* is a callable that runs dynamics for a given trajectory segment.
Its takes a :class:`~westpa.Segment` object as input, reads its ``initpoint``,
sets its ``endpoint``, and returns the modified object.

As a simple example, the following function is a propagator for a symmetric random walk::

    import random

    def propagate(segment):
        segment.endpoint = segment.initpoint + random.choice([-1, 1])
        return segment

WESTPA provides a built-in propagator type, :class:`~westpa.OpenMMPropagator`,
for running molecular dynamics with the `OpenMM <https://openmm.org>`_ toolkit.
To use it, OpenMM must be installed; this can be done either directly:

.. code-block::

    python -m pip install openmm

or by installing WESTPA with the ``openmm`` extra:

.. code-block::

    python -m pip install westpa[openmm]

.. autoclass:: westpa.OpenMMPropagator

.. autoclass:: westpa.OpenMMReport
