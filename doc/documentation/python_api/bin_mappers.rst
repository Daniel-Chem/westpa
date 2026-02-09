Bin mappers
===========

A *bin mapper* defines a method for grouping trajectory segments into bins.

WESTPA provides several built-in bin mapper types:

.. autosummary::
   :nosignatures:

   ~westpa.FuncBinMapper
   ~westpa.MABBinMapper
   ~westpa.PiecewiseBinMapper
   ~westpa.RectilinearBinMapper
   ~westpa.RecursiveBinMapper
   ~westpa.VectorizingFuncBinMapper
   ~westpa.VoronoiBinMapper

Custom bin mapper types may be defined by subclassing the
:class:`~westpa.BinMapper` base class.

.. autoclass:: westpa.BinMapper
   :members: nbins, map

.. autoclass:: westpa.Bin()
   :members: bisect_weights, reweight

.. autoclass:: westpa.FuncBinMapper
.. autoclass:: westpa.MABBinMapper
.. autoclass:: westpa.PiecewiseBinMapper
.. autoclass:: westpa.RectilinearBinMapper
.. autoclass:: westpa.RecursiveBinMapper
.. autoclass:: westpa.VectorizingFuncBinMapper
.. autoclass:: westpa.VoronoiBinMapper
