from collections.abc import Iterable, Sequence
from typing import Protocol

from .core.segment import Segment


class Propagator(Protocol):
    """Callback protocol for propagators."""

    def __call__(self, segments: Sequence[Segment]) -> Iterable[Segment]:
        """Propagate a batch of trajectory segments.

        Parameters
        ----------
        segments : sequence of Segment
            Segments to propagate.

        Returns
        -------
        segments : iterable of Segment
            Propagated segments.

        """
        ...
