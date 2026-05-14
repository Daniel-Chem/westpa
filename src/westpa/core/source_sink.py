from collections.abc import Container

import numpy as np

from .state import State


class Source:
    """Set of states from which trajectories are re-initiated upon reaching a sink.

    Parameters
    ----------
    states : iterable of :class:`State`
        Set of source states.
    p : 1-D array-like, optional
        Selection probability of each state. Defaults to a uniform distribution.

    Examples
    --------

    >>> import westpa
    >>> states = [westpa.State(coord=[0.]), westpa.State(coord=[1.])]

    Uniform selection probability:

    >>> westpa.Source(states)
    Source([State(coord=array([0.])), State(coord=array([1.]))], p=[0.5, 0.5])

    Non-uniform selection probability:

    >>> westpa.Source(states, p=[0.7, 0.3])
    Source([State(coord=array([0.])), State(coord=array([1.]))], p=[0.7, 0.3])

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

    def random_choice(self, k=1, rng=None):
        """Randomly sample states from the source distribution.

        Parameters
        ----------
        k : int, optional
            Sample size.
        rng : numpy.random.Generator, optional
            Pseudo-random number generator to use. Defaults to
            ``numpy.random.default_rng()``.

        Returns
        -------
        sample : iterable of :class:`State`
            Sample of `k` source states.

        """
        rng = np.random.default_rng(rng)
        return rng.choice(self.states, p=self.p, size=k).tolist()


class Sink(Container):
    """Describes a sink (target) set.

    Parameters
    ----------
    indicator : Callable[[:class:`Segment`], bool]
        Function that returns True if a given trajectory segment reached the
        sink, False otherwise.
    label : str, optional
        Descriptive label for the sink.

    Examples
    --------

    Create a sink containing segments with final (-1), first-dimension
    (0) progress coordinate values greater than 1.0:

    >>> import westpa
    >>> sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)

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
