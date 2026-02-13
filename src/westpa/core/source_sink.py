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
        Probability to assign each state. Defaults to a uniform distribution.

    Attributes
    ----------
    states : sequence of State
        Source states.
    p : numpy.ndarray
        Probability assigned to each state.

    Examples
    --------
    Single state:

    >>> import westpa
    >>> westpa.Source(westpa.State(label='a'))
    Source([State(label='a')], p=[1.0])

    Two states with equal probabilities:

    >>> states = [westpa.State(label='a'), westpa.State(label='b')]
    >>> westpa.Source(states)
    Source([State(label='a'), State(label='b')], p=[0.5, 0.5])

    Two states with different probabilities:

    >>> westpa.Source(states, p=[0.7, 0.3])
    Source([State(label='a'), State(label='b')], p=[0.7, 0.3])


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
                raise ValueError("length of 'p' must match number of states")
        p /= p.sum()

        self.states = states
        self.p = p

    def __repr__(self):
        args = f'{self.states}, p={self.p.tolist()}'
        return type(self).__name__ + '(' + args + ')'

    def random_choice(self, k=1, seed=None):
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

    Attributes
    ----------
    indicator : Callable[[Segment], bool]
        Indicator function.

    Methods
    -------
    __contains__

    Examples
    --------
    Create a sink containing segments with final (``-1``), first-dimension
    (``0``) progress coordinate values greater than one:

    >>> import westpa
    >>> sink = westpa.Sink(lambda segment: segment.pcoord[-1, 0] > 1.0)

    Test for membership:

    >>> westpa.Segment(pcoord=[[0.9]]) in sink
    False
    >>> westpa.Segment(pcoord=[[1.1]]) in sink
    True

    """

    def __init__(self, indicator):
        self.indicator = indicator

    def __contains__(self, segment):
        """Test whether a given segment is contained in the sink."""
        return self.indicator(segment)

    def __repr__(self):
        args = f'indicator={self.indicator!r}'
        return type(self).__name__ + '(' + args + ')'
