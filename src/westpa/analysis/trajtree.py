from ..core._data_manager import DataManager  # noqa


class TrajectoryTree:
    """Interface for analyzing the trajectory data from a simulation.

    Parameters
    ----------
    datafile : str or io.BufferedIOBase
        HDF5 file containing simulation data.
    load_pcoords : bool, default True
        Whether to load progress coordinates when retrieving trajectory
        segment data.

    Attributes
    ----------
    datafile : str or io.BufferedIOBase
    load_pcoords : bool

    Methods
    -------
    open_datafile
    close_datafile
    get_segment
    get_segments
    parent
    trace

    Examples
    --------

    >>> import westpa
    >>> trajtree = westpa.TrajectoryTree('west.h5')

    Open the HDF5 file for reading:

    >>> trajtree.open_datafile()

    Retrieve a single segment:

    >>> trajtree.get_segment(1, 0)
    <Segment n_iter=1, seg_id=0, weight=0.2, parent_id=-1, wtg_parent_ids=(-1,) at 0x1694df380>

    Retrieve multiple segments from a given iteration:

    >>> trajtree.get_segments(5, [13, 9, 17])
    [<Segment n_iter=5, seg_id=13, weight=0.02, parent_id=4, wtg_parent_ids=(3, 4) at 0x1632a36e0>,
     <Segment n_iter=5, seg_id=9, weight=0.01, parent_id=7, wtg_parent_ids=(7,) at 0x1632a3680>,
     <Segment n_iter=5, seg_id=17, weight=0.02, parent_id=5, wtg_parent_ids=(5,) at 0x1632a35f0>]

    Get the parent of a segment:

    >>> segment = trajtree.get_segment(5, 9)
    >>> trajtree.parent(segment)
    <Segment n_iter=4, seg_id=7, weight=0.02, parent_id=14, wtg_parent_ids=(14,) at 0x179e445c0>

    Trace the lineage of a segment:

    >>> trajtree.trace(segment)
    [<Segment n_iter=1, seg_id=1, weight=0.2, parent_id=-2, wtg_parent_ids=(-2,) at 0x16bcb6b70>,
     <Segment n_iter=2, seg_id=13, weight=0.04, parent_id=1, wtg_parent_ids=(1,) at 0x16bcb68a0>,
     <Segment n_iter=3, seg_id=14, weight=0.08, parent_id=13, wtg_parent_ids=(10, 13) at 0x16bcb6930>,
     <Segment n_iter=4, seg_id=7, weight=0.02, parent_id=14, wtg_parent_ids=(14,) at 0x16bcb6270>,
     <Segment n_iter=5, seg_id=9, weight=0.01, parent_id=7, wtg_parent_ids=(7,) at 0x16bb8be30>]

    Close the HDF5 file:

    >>> trajtree.close_datafile()

    """

    def __init__(
        self,
        datafile,
        *,
        load_pcoords=True,
    ):
        self.data_manager = DataManager(datafile)
        self._load_pcoords = None
        self.load_pcoords = load_pcoords

    @property
    def datafile(self):
        """HDF5 data file."""
        return self.data_manager.we_h5filename

    @property
    def load_pcoords(self):
        """Whether to load progress coordinates."""
        return self._load_pcoords

    @load_pcoords.setter
    def load_pcoords(self, value):
        if not isinstance(value, bool):
            raise TypeError("'load_pcoords' must be True or False")
        self._load_pcoords = value

    def open_datafile(self):
        """Open the HDF5 file for reading."""
        self.data_manager.open_backing(mode='r')

    def close_datafile(self):
        """Close the HDF5 file."""
        self.data_manager.close_backing()

    def __enter__(self):
        self.open_datafile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_datafile()
        return False

    def get_segments(self, n_iter, seg_ids=None):
        """Retrieve multiple segments from a given iteration.

        Parameters
        ----------
        n_iter : int
            Iteration number.
        seg_ids : list of int, optional
            Segment indices. If not provided, all the segments from iteration
            `n_iter` will be returned.

        Returns
        -------
        segments : list of :class:`Segment`
            Selected segments.

        """
        if not isinstance(n_iter, int):
            raise TypeError("'n_iter' must be an integer")
        if n_iter not in range(1, iter_stop := self.data_manager.current_iteration):
            raise ValueError(f'iteration number must be in range(1, {iter_stop})')

        iter_group = self.data_manager.get_iter_group(n_iter)
        n_total_segments = iter_group['seg_index'].shape[0]

        if seg_ids is not None:
            if not all(isinstance(seg_id, int) for seg_id in seg_ids):
                raise TypeError("'seg_ids' must be a list of integers")
            for seg_id in seg_ids:
                if seg_id not in range(n_total_segments):
                    raise ValueError(f'segment index {seg_id} out of range for iteration {n_iter}')

        segments = self.data_manager.get_segments(n_iter, seg_ids, load_pcoords=self.load_pcoords)

        if seg_ids is not None:
            # reorder segments to match order of indices in `seg_ids`
            segments_by_id = {s.seg_id: s for s in segments}
            segments = [segments_by_id[seg_id] for seg_id in seg_ids]

        return segments

    def get_segment(self, n_iter, seg_id):
        """Retrieve a single segment.

        Parameters
        ----------
        n_iter : int
            Iteration number.
        seg_id : int
            Segment index.

        Returns
        -------
        segment : :class:`Segment`
            Selected segment.

        """
        return self.get_segments(n_iter, seg_ids=[seg_id])[0]

    def get_parent_ids(self, n_iter, seg_ids=None):
        return self.data_manager.get_parent_ids(n_iter, seg_ids)

    def parent(self, segment):
        """Return the parent of a segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to find the parent of.

        Returns
        -------
        parent : :class:`Segment` or None
            Parent of the given segment, or None if the segment has no parent.

        """
        if segment.initpoint_type == segment.InitPointType.NEWTRAJ:
            return None
        else:
            return self.get_segment(segment.n_iter - 1, segment.parent_id)

    def trace(self, segment):
        """Trace the lineage of a segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to trace.

        Returns
        -------
        trace : sequence of :class:`Segment`
            Sequence of segments in the trajectory leading up to and including
            `segment`.

        """
        segments = [segment]
        while parent := self.parent(segment):
            segments.append(parent)
            segment = parent
        segments.reverse()
        return segments
