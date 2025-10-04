Resamplers
==========

A *resampler* is a callable that takes a set of propagated trajectory segments
as input and returns a set of resampled segments. WESTPA provides one built-in
resampler type: :class:`~westpa.HuberKimResampler`. Custom resamplers can be
implemented using the :func:`~westpa.split` and :func:`~westpa.merge` functions.

.. autoclass:: westpa.HuberKimResampler

.. autofunction:: westpa.split

.. autofunction:: westpa.merge
