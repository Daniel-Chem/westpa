Propagators
===========

.. autoclass:: westpa.Propagator
   :members:

WESTPA provides a built-in propagator type, :class:`~westpa.OpenMMPropagator`,
for running molecular dynamics with the `OpenMM <https://openmm.org>`_ toolkit.
To use it, OpenMM must be installed, e.g.:

.. code-block:: shell

    pip install "openmm[cuda12]"

.. autoclass:: westpa.OpenMMPropagator
   :members:
