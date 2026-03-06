from collections.abc import Container

import numpy as np

from .state import State


class Source:
    """Represents a source (initial) distribution.

    Parameters
    ----------
    states : State or iterable of State
        One or more source states.
    p : 1-D array-like, optional
        Selection probability of each state. Defaults to a uniform distribution.

    Examples
    --------

    >>> import westpa
    >>> states = [westpa.State(coord=[0.0]), westpa.State(coord=[1.0])]

    Uniform selection probabilities (default):

    >>> westpa.Source(states)
    Source([State(coord=[0.0]), State(coord=[1.0])], p=[0.5, 0.5])

    Non-uniform selection probabilities:

    >>> westpa.Source(states, p=[0.7, 0.3])
    Source([State(coord=[0.0]), State(coord=[1.0])], p=[0.7, 0.3])

    """

    def __init__(self, states, p=None):
        if isinstance(states, State):
            states = [states]
        else:
            states = list(states)
            if not all(isinstance(item, State) for item in states):
                raise TypeError("items in 'states' must be westpa.State objects")

        if p is None:
            p = np.ones(len(states))
        else:
            p = np.asarray(p, dtype=float)
            if len(p) != len(states):
                raise ValueError("length of 'p' must match the number of states")
        p /= p.sum()

        self.states = states
        self.p = p

    def __repr__(self):
        args = f'{self.states}, p={self.p.tolist()}'
        return type(self).__name__ + '(' + args + ')'

    def random_choice(self, k=1, seed=None):
        """Generate a random sample of source states.

        Parameters
        ----------
        k : int, optional
            Sample size.
        seed : int, sequence of int, or numpy.random.Generator, optional
            Seed to pass to ``numpy.random.default_rng()``. If a ``Generator`` is
            passed, it will be used directly.

        Returns
        -------
        iterable of State
            Generated sample of `k` states.

        """
        rng = np.random.default_rng(seed)
        return rng.choice(self.states, p=self.p, size=k).tolist()


class Sink(Container):
    """Represents a sink (target) region.

    Parameters
    ----------
    indicator : Callable[[Segment], bool]
        Function that returns True if a given trajectory segment reached the
        sink, False otherwise. This function may assume that the segment has
        completed propagation and that its ``pcoord`` attribute has been set.
    label : str, optional
        Descriptive label for the sink.

    Examples
    --------
    Create a sink containing segments with final (``-1``), first-dimension
    (``0``) progress coordinate values greater than one:

    >>> import westpa
    >>> sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)

    Test for membership:

    >>> westpa.Segment(pcoord=[[0.9]]) in sink
    False
    >>> westpa.Segment(pcoord=[[1.1]]) in sink
    True

    """

    def __init__(self, indicator, label=None):
        self.indicator = indicator
        self.label = label or ''

    def __contains__(self, segment):
        """Test whether a given segment is contained in the sink."""
        return self.indicator(segment)

    def __repr__(self):
        args = f'indicator={self.indicator!r}'
        if self.label:
            args += f', label={self.label!r}'
        return type(self).__name__ + '(' + args + ')'
