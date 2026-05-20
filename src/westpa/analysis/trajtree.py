from collections.abc import Sequence
from itertools import chain
from typing import NamedTuple

import networkx as nx

from ..core._data_manager import DataManager  # noqa


class Node(NamedTuple):
    n_iter: int
    seg_id: int


class TrajectoryTree:
    """Interface for analyzing WE trajectory data.

    Parameters
    ----------
    datafile : str or io.BufferedIOBase
        HDF5 file containing simulation data.
    load_pcoords : bool, default True
        Whether to load progress coordinates when retrieving trajectory
        segments.

    Attributes
    ----------
    datafile : str or io.BufferedIOBase
    load_pcoords : bool
    n_iters : int

    Methods
    -------
    segment_count
    get_segment
    get_segments
    get_parent_ids
    parent
    children
    trace
    close

    Examples
    --------

    >>> import westpa
    >>> trajtree = westpa.TrajectoryTree('west.h5')

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

    >>> traj = trajtree.trace(segment)
    >>> traj
    <Trajectory with 5 segments at 0x15c8b0500>
    >>> list(traj)
    [<Segment n_iter=1, seg_id=1, weight=0.2, parent_id=-2, wtg_parent_ids=(-2,) at 0x16bcb6b70>,
     <Segment n_iter=2, seg_id=13, weight=0.04, parent_id=1, wtg_parent_ids=(1,) at 0x16bcb68a0>,
     <Segment n_iter=3, seg_id=14, weight=0.08, parent_id=13, wtg_parent_ids=(10, 13) at 0x16bcb6930>,
     <Segment n_iter=4, seg_id=7, weight=0.02, parent_id=14, wtg_parent_ids=(14,) at 0x16bcb6270>,
     <Segment n_iter=5, seg_id=9, weight=0.01, parent_id=7, wtg_parent_ids=(7,) at 0x16bb8be30>]

    Close the HDF5 file:

    >>> trajtree.close()

    Use as a context manager to close the HDF5 file automatically:

    >>> with westpa.TrajectoryTree('west.h5') as trajtree:
    ...     segment = trajtree.get_segment(5, 9)
    ...
    >>> segment
    <Segment n_iter=5, seg_id=9, weight=0.01, parent_id=7, wtg_parent_ids=(7,) at 0x161da3b60>

    """

    def __init__(
        self,
        datafile,
        load_pcoords=True,
    ):
        self.data_manager = DataManager(datafile)
        self.data_manager.open_backing(mode='r')

        self._load_pcoords = None
        self.load_pcoords = load_pcoords

        # Construct a DiGraph to allow traversal without hitting the HDF5 file.
        # Nodes are (n_iter, seg_id) pairs. Edges point from parent to child.
        graph = nx.DiGraph()
        for n_iter in range(1, self.n_iters + 1):
            nodes = [Node(n_iter, seg_id) for seg_id in range(self.segment_count(n_iter))]
            graph.add_nodes_from(nodes, subset=n_iter)
            if n_iter == 1:
                continue
            parent_ids = self.data_manager.get_all_parent_ids(n_iter).tolist()
            for node, parent_id in zip(nodes, parent_ids):
                if parent_id < 0:
                    return
                graph.add_edge(Node(n_iter - 1, parent_id), node)

        self._graph = graph

    def close(self):
        """Close the HDF5 file."""
        self.data_manager.close_backing()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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

    @property
    def n_iters(self):
        """Number of iterations (layers) in the trajectory tree."""
        return self.data_manager.current_iteration - 1

    def _get_iter_group(self, n_iter):
        if not isinstance(n_iter, int):
            raise TypeError("'n_iter' must be an integer")
        if n_iter not in range(1, iter_stop := self.data_manager.current_iteration):
            raise ValueError(f"'n_iter' must be in range(1, {iter_stop})")
        return self.data_manager.get_iter_group(n_iter)

    def segment_count(self, n_iter=None):
        """Return the total number of segments in a given iteration or in the
        entire trajectory tree.

        Parameters
        ----------
        n_iter : int, optional
            Iteration number. If provided, the total number of segments in the
            given iteration will be returned. If None (the default), the total
            number of segments in the trajectory tree will be returned.

        Returns
        -------
        n_segments : int
            Total number of segments in the given iteration if `n_iter` was
            provided; else the total number of segments in the trajectory tree.

        """
        if n_iter is None:
            return self._history_graph.number_of_nodes()
        else:
            return self._get_iter_group(n_iter)['seg_index'].shape[0]

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
        if seg_ids is not None:
            if not all(isinstance(seg_id, int) for seg_id in seg_ids):
                raise TypeError("'seg_ids' must be a list of integers")
            n_segments = self.segment_count(n_iter)
            for seg_id in seg_ids:
                if seg_id not in range(n_segments):
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
        """Return the parent indices of all or selected segments from a given iteration.

        Parameters
        ----------
        n_iter : int
            Iteration number.
        seg_ids : list of int, optional
            Indices of the segments for which to retrieve parent indices.
            If not provided, the parent indices of all the segments (in order)
            will be returned.

        Returns
        -------
        parent_ids : list of int
            Parent indices of the selected segments.

        """
        return self.data_manager.get_parent_ids(n_iter, seg_ids)

    def parent(self, segment, n=1):
        """Return the `n`-th level parent of a segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to find the parent of.
        n : int, default 1
            Number of iterations by which the parent is removed from `segment`
            (1 = direct parent, 2 = grandparent, etc.).

        Returns
        -------
        parent : :class:`Segment` or None
            `n`-th level parent of the given segment, or None if the segment
            has no `n`-th level parent.

        """
        if not isinstance(n, int):
            raise TypeError("'n' must be an integer")
        if n < 1:
            raise ValueError("'n' must be greater than or equal to 1")

        node = Node(segment.n_iter, segment.seg_id)

        for _ in range(n):
            predecessor = next(self._graph.predecessors(node), None)
            if predecessor is None:
                return None
            node = predecessor

        return self.get_segment(node.n_iter, node.seg_id)

    def children(self, segment, n=1):
        """Return the `n`-th level children of a segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to find the children of.
        n : int, default 1
            Number of iterations by which the children are removed from
            `segment` (1 = direct children, 2 = grandchildren, etc.).

        Returns
        -------
        children : list of :class:`Segment`
            `n`-th level children of the given segment (empty if the
            segment has no `n`-th level children).

        """
        if not isinstance(n, int):
            raise TypeError("'n' must be an integer")
        if n < 1:
            raise ValueError("'n' must be greater than or equal to 1")

        nodes = [Node(segment.n_iter, segment.seg_id)]

        for _ in range(n):
            successors = list(chain.from_iterable(map(self._graph.successors, nodes)))
            if not successors:
                return []
            nodes = successors

        return self.get_segments(nodes[0].n_iter, [node.seg_id for node in nodes])

    def trace(self, segment, maxlen=None):
        """Trace the lineage of a segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to trace.
        maxlen : int, optional
            Maximum number of segments in the returned trajectory trace.

        Returns
        -------
        traj : :class:`Trajectory`
            Trajectory leading up to and including `segment`.

        """
        if maxlen is not None:
            if not isinstance(maxlen, int):
                raise TypeError("'maxlen' must be an integer")
            if maxlen < 1:
                raise ValueError("'maxlen' must be positive")

        segments = [segment]
        while parent := self.parent(segment):
            if maxlen is not None and len(segments) == maxlen:
                break
            segments.append(parent)
            segment = parent

        return Trajectory(reversed(segments))

    def __repr__(self):
        return f'<{type(self).__name__} with {self.n_iters} iterations at {hex(id(self))}>'


class Trajectory(Sequence):
    """A contiguous sequence of trajectory segments.

    Parameters
    ----------
    segments : iterable of :class:`Segment`
        Segments comprising the trajectory.

    Attributes
    ----------
    states : iterator of :class:`State`
    initial_state : :class:`State`
    final_state : :class:`State`
    iter_range : range

    """

    def __init__(self, segments):
        self.segments = tuple(segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    @property
    def states(self):
        """Sequence of states visited by the trajectory."""
        for segment in self.segments:
            yield segment.initial_state
        yield self.segments[-1].final_state

    @property
    def initial_state(self):
        """Initial state of the trajectory."""
        return self.segments[0].initial_state

    @property
    def final_state(self):
        """Final state of the trajectory."""
        return self.segments[-1].final_state

    @property
    def iter_range(self):
        """Range of iterations spanned by the trajectory."""
        return range(self.segments[0].n_iter, self.segments[-1].n_iter + 1)

    def __repr__(self):
        return f'<{type(self).__name__} with {len(self)} segments at {hex(id(self))}>'
