Resamplers
==========

A *resampler* is a callable that takes a set of propagated trajectory segments,
performs weighted ensemble resampling (e.g., using the :func:`~westpa.split` and
:func:`~westpa.merge` functions), and returns the set of resampled segments.

WESTPA provides one built-in resampler type: :class:`~westpa.HuberKimResampler`.

.. autoclass:: westpa.HuberKimResampler

.. autofunction:: westpa.split

.. autofunction:: westpa.merge
