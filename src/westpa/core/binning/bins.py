import logging
import operator
from collections.abc import MutableSet

import numpy as np
from sortedcontainers import SortedSet

logger = logging.getLogger(__name__)

EPS = np.finfo(np.float64).eps


# Under the hood,
class Bin(MutableSet):
    """A mutable, sorted set of trajectory segments.

    The segments in a ``Bin`` are sorted in increasing order of weight, and
    the order is automatically maintained as the set is updated.

    Attributes
    ----------
    label : str
    count : int
    weight : float
    segments : sequence of Segment
    weights : numpy.ndarray

    """

    def __init__(self, iterable=None, label=None):
        self._segments = SortedSet(iterable, key=operator.attrgetter('weight'))
        self._label = label or ''

    def __repr__(self):
        names = ['label', 'count', 'weight']
        attrs = ', '.join(f'{name}={getattr(self, name)!r}' for name in names)
        return f'<{self.__class__.__name__} at {hex(id(self))}, {attrs}>'

    # The next five methods are required to implement MutableSet.
    def __contains__(self, item):
        return item in self._segments

    def __iter__(self):
        return iter(self._segments)

    def __len__(self):
        return len(self._segments)

    def add(self, elem):
        self._segments.add(elem)

    def discard(self, elem):
        self._segments.discard(elem)

    # MutableSet doesn't provide update() or difference_update().
    def update(self, *others):
        self._segments.update(*others)

    def difference_update(self, *others):
        self._segments.difference_update(*others)

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

    @property
    def segments(self):
        """Segments in the bin, sorted in increasing order of weight."""
        return self._segments

    @property
    def weights(self):
        """Segment weights, sorted in increasing order."""
        return np.array([segment.weight for segment in self])

    def bisect_weights(self, w, side='left'):
        """Find the index where `w` should be inserted in :attr:`weights` to maintain sorted order.

        Parameters
        ----------
        w : float
            Value to insert.
        side : {'left', 'right'}, optional
            If 'left', the returned index ``i`` satisfies ``weights[i-1] < w <= weights[i]``.
            If 'right', it satisfies ``weights[i-1] <= w < weights[i]``.

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
            New total weight of the bin.

        """
        if len(self) == 0 and new_weight == 0:
            return

        if len(self) == 0 and new_weight != 0:
            raise ValueError('cannot reweight empty bin')

        current_weight = self.weight
        logger.debug('reweighting bin with {:d} segments from {:g} to {:g}'.format(len(self), current_weight, new_weight))
        assert (new_weight == 0 and current_weight == 0) or new_weight > 0

        wrat = new_weight / current_weight
        for p in self:
            p.weight *= wrat

        logger.debug('new weight: {:g}'.format(self.weight))
        assert abs(new_weight - self.weight) <= EPS * len(self)
