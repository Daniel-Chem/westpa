import abc

from westpa.core.segment import Segment


class Propagator(abc.ABC):
    """Abstract base class for propagators."""

    @abc.abstractmethod
    def __call__(self, segment: Segment) -> Segment:
        """Run dynamics for a given segment."""
        segment.endpoint = segment.initpoint  # no operation
        return segment
