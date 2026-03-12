Propagators
===========

.. autoclass:: westpa.Propagator
   :members: propagate

.. autoclass:: westpa.BatchedPropagator
   :members: propagate_batch

WESTPA provides a built-in propagator type, :class:`~westpa.OpenMMPropagator`,
for running molecular dynamics with the `OpenMM <https://openmm.org>`_ toolkit.
To use it, OpenMM must be installed, e.g.:

.. code-block:: shell

    pip install "openmm[cuda12]"

.. autoclass:: westpa.OpenMMPropagator
.. autoclass:: westpa.OpenMMReport
