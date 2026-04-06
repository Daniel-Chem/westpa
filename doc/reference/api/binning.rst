Binning
=======

.. autosummary::
   :nosignatures:

   ~westpa.VoronoiBinMapper
   ~westpa.RecursiveBinMapper
   ~westpa.RectilinearBinMapper
   ~westpa.MABBinMapper
   ~westpa.BinMapper
   ~westpa.Bin

Custom bin mapper types may be defined by subclassing the
:class:`~westpa.BinMapper` base class.

.. autoclass:: westpa.VoronoiBinMapper
.. autoclass:: westpa.RecursiveBinMapper
.. autoclass:: westpa.RectilinearBinMapper
.. autoclass:: westpa.MABBinMapper

.. autoclass:: westpa.BinMapper
   :members: nbins, map

.. autoclass:: westpa.Bin
   :show-inheritance:
   :members:
