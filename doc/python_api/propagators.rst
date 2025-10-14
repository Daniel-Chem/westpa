Propagators
===========

A *propagator* is a callable that runs dynamics for a given trajectory segment.
It takes a :class:`~westpa.Segment` object as input, reads its initial state,
sets its final state, and returns the modified object.

To illustrate, the following function is a propagator for a symmetric random
walk on the integers::

    import random
    import westpa

    def propagate(segment, steps=100):
        s = segment.initial_state.coord.item()
        for _ in range(steps):
            s += random.choice([-1, 1])
        segment.final_state = westpa.State(coord=[s])
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
