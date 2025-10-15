import abc
from collections.abc import Iterable

from westpa.core.segment import Segment


class Resampler(abc.ABC):
    """Abstract base class for resamplers."""

    @abc.abstractmethod
    def __call__(self, segments: Iterable[Segment]) -> Iterable[Segment]:
        """Resample a set of trajectory segments."""
        return segments  # no operation
