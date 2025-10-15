import numpy as np

from ..segment import Segment


def split(segment, m=2):
    """Split a trajectory segment into two or more copies.

    Parameters
    ----------
    segment : Segment
        Segment to split.
    m : int, default 2
        Number of copies to create.

    Returns
    -------
    list of Segment
        Segments created by splitting `segment` into `m` equally weighted copies.

    """
    if not isinstance(m, int):
        raise TypeError("'m' must be an integer")
    if not m >= 2:
        raise ValueError("'m' must be greater than or equal to 2")
    new_weight = segment.weight / m
    return [
        Segment(
            n_iter=segment.n_iter,
            seg_id=segment.seg_id,
            weight=new_weight,
            parent_id=segment.parent_id,
            wtg_parent_ids=segment.wtg_parent_ids,
            initial_state=segment.initial_state,
            final_state=segment.final_state,
            pcoord=segment.pcoord,
            data=segment.data,
        )
        for _ in range(m)
    ]


def merge(segments, seed=None):
    """Merge multiple trajectory segments into a single segment. The surviving
    segment is chosen randomly according to weight.

    Parameters
    ----------
    segments : iterable of Segment
        Segments to merge.
    seed : int or sequence of int, optional
        Seed to initialize the random state.

    Returns
    -------
    Segment
        Copy of the surviving segment, assigned the total weight of `segments`.

    """
    segments = list(segments)
    weights = np.array([segment.weight for segment in segments])
    total_weight = weights.sum()
    rng = np.random.default_rng(seed)
    choice = rng.choice(segments, p=weights / total_weight)
    return Segment(
        n_iter=segments[0].n_iter,
        seg_id=choice.seg_id,
        weight=total_weight,
        parent_id=choice.parent_id,
        wtg_parent_ids=set.union(*(segment.wtg_parent_ids for segment in segments)),
        initial_state=choice.initial_state,
        final_state=choice.final_state,
        pcoord=choice.pcoord,
        data=choice.data,
    )
