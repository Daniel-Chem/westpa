Resamplers
==========

A *resampler* defines a method for resampling (e.g., splitting and merging)
the trajectory segments in a given bin.

WESTPA provides three built-in resampler types:

.. autosummary::
   :nosignatures:

   ~westpa.HuberKimResampler
   ~westpa.MultinomialResampler
   ~westpa.ResidualResampler

Custom resampler types may be defined by subclassing the
:class:`~westpa.Resampler` base class.

.. autoclass:: westpa.Resampler
   :members: resample, split, merge, __call__

.. autoclass:: westpa.HuberKimResampler
.. autoclass:: westpa.MultinomialResampler
.. autoclass:: westpa.ResidualResampler
