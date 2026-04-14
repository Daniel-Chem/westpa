from .base import Resampler
from .ops import resample_equal_weight


class MultinomialResampler(Resampler):
    """Randomly samples trajectories according to their relative weights.

    Parameters
    ----------
    seed, smallest_allowed_weight, largest_allowed_weight
        See :class:`Resampler` class documentation for details.

    """

    def resample(self, bin, target_count):
        weights = bin.weights()
        total_weight = weights.sum()

        counts = self.rng.multinomial(n=target_count, pvals=weights / total_weight)

        ideal_weight = total_weight / target_count
        return resample_equal_weight(bin, counts, new_weight=ideal_weight)
