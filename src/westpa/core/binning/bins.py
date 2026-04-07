import logging
import operator
from collections.abc import MutableSet

import numpy as np
from sortedcontainers import SortedSet

logger = logging.getLogger(__name__)

EPS = np.finfo(np.float64).eps


# Under the hood,
class Bin(MutableSet):
    """A mutable, sorted set of :class:`Segment` objects. The segments in a bin
    are sorted in increasing order of weight, and the order is automatically
    maintained as the bin is updated.

    Parameters
    ----------
    segments : iterable of Segment, optional
        Segments to add to the bin.
    label : str, optional
        Bin label.

    Attributes
    ----------
    label : str or None
    count : int
    weight : float

    Methods
    -------
    __contains__
    __getitem__
    __iter__
    __len__
    add
    discard
    weights
    bisect_weights

    """

    def __init__(self, segments=None, label=None):
        self._segments = SortedSet(segments, key=operator.attrgetter('weight'))
        self._label = label

    def __repr__(self):
        names = ['label', 'count', 'weight']
        attrs = ', '.join(f'{name}={getattr(self, name)!r}' for name in names)
        return f'<{self.__class__.__name__} at {hex(id(self))}, {attrs}>'

    # The next five methods are required to implement MutableSet.
    def __contains__(self, segment):
        """Return True if the segment is in the bin, False otherwise."""
        return segment in self._segments

    def __iter__(self):
        """Return an iterator over the segments in the bin."""
        return iter(self._segments)

    def __len__(self):
        """Return the number of segments in the bin."""
        return len(self._segments)

    def add(self, segment):
        """Add a segment to the bin."""
        self._segments.add(segment)

    def discard(self, segment):
        """Remove a segment from the bin. Do not raise an exception if absent."""
        self._segments.discard(segment)

    # MutableSet doesn't provide update() or difference_update(), used by we_driver.
    def update(self, *others):
        self._segments.update(*others)

    def difference_update(self, *others):
        self._segments.difference_update(*others)

    # Implement Sequence.
    def __getitem__(self, index):
        """Return the segment at the given index. Supports slicing."""
        return self._segments[index]

    @property
    def label(self):
        """Bin label."""
        return self._label

    @property
    def count(self):
        """Number of segments in the bin."""
        return len(self)

    @property
    def weight(self):
        """Total weight of all segments in the bin."""
        return sum(map(self._segments.key, self))

    def weights(self):
        """Return the segment weights, sorted in increasing order.

        Returns
        -------
        numpy.ndarray
            Sorted array of weights.

        """
        return np.array(list(map(self._segments.key, self)))

    def bisect_weights(self, w, side='left'):
        """Find the index where `w` should be inserted in ``self.weights()`` to maintain sorted order.

        Parameters
        ----------
        w : float
            Value to insert.
        side : {'left', 'right'}, optional
            If 'left', return the insertion point before (to the left of) any
            existing entries of `w`. If 'right', return the insertion point
            after (to the right of) any existing entries of `w`.

        Returns
        -------
        int
            Insertion point for `w`.

        """
        match side:
            case 'left':
                return self._segments.bisect_key_left(w)
            case 'right':
                return self._segments.bisect_key_right(w)
            case _:
                raise ValueError("'side' must be either 'left' or 'right'")

    def reweight(self, new_weight):
        """Reweight the bin by scaling the segment weights.

        Parameters
        ----------
        new_weight : float
            New :attr:`weight` of the bin after reweighting. Must be between 0 and 1.

        """
        if not (0 <= new_weight <= 1):
            raise ValueError("'new_weight' must be between 0 and 1")

        if len(self) == 0:
            if new_weight > 0:
                raise ValueError('cannot reweight empty bin')
            return

        ratio = new_weight / self.weight
        segments = [segment.replace(weight=ratio * segment.weight) for segment in self]
        self._segments = SortedSet(segments, key=operator.attrgetter('weight'))
