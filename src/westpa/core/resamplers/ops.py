import operator

import numpy as np


def split(segment, bin, m=2):
    """Split a trajectory segment into two or more copies.

    Parameters
    ----------
    segment : :class:`Segment`
        Segment to split.
    bin : :class:`Bin`
        Bin the segment belongs to. This function modifies the contents of `bin`.
    m : int, default 2
        Number of copies to split `segment` into.

    """
    if not isinstance(m, int):
        raise TypeError("'m' must be an integer")
    if not m >= 2:
        raise ValueError("'m' must be greater than or equal to 2")

    new_weight = segment.weight / m
    new_segments = {segment.replace(weight=new_weight) for _ in range(m)}

    bin.remove(segment)
    bin |= new_segments


def merge(segments, bin, total_weight=None, rng=None):
    """Merge multiple trajectory segments into a single segment. The
    surviving segment is chosen randomly according to weight.

    Parameters
    ----------
    segments : iterable of :class:`Segment`
        Segments to merge.
    bin : :class:`Bin`
        Bin the segments belong to. This function modifies the contents of `bin`.
    total_weight : float, optional
        Combined weight of `segments`. If not passed, the value will be
        computed by this function.
    rng : numpy.random.Generator, optional
        Pseudo-random number generator to use. Defaults to
        ``numpy.random.default_rng()``.

    """
    segments = list(segments)
    weights = np.array(list(map(operator.attrgetter('weight'), segments)))

    if total_weight is None:
        total_weight = weights.sum()

    rng = np.random.default_rng(rng)
    segment = rng.choice(segments, p=weights / total_weight)
    new_segment = segment.replace(
        weight=total_weight,
        wtg_parent_ids=set.union(*(segment.wtg_parent_ids for segment in segments)),
    )

    bin -= segments
    bin.add(new_segment)


def resample_equal_weight(bin, counts, new_weight=None):
    """Create an equal-weight sample of trajectory segments in a given bin.

    Parameters
    ----------
    bin : :class:`Bin`
        Bin to resample.
    counts : array_like of int
        Number of copies of each trajectory segment in `bin` to include
        in the sample: ``counts[i]`` is the number of copies of segment
        ``bin[i]``.
    new_weight : float, optional
        Weight to assign each trajectory in the new sample. Defaults to
        ``bin.weight / sum(counts)``.

    Returns
    -------
    Bin
        Resampled bin.

    """
    counts = np.asarray(counts)

    if new_weight is None:
        new_weight = bin.weight / counts.sum()

    wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))

    new_segments = set()
    for segment, count in zip(bin, counts):
        for _ in range(count):
            new_segment = segment.replace(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
            new_segments.add(new_segment)

    bin.clear()
    bin |= new_segments

    return bin
