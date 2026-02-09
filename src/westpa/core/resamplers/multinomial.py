from .base import Resampler


class MultinomialResampler(Resampler):
    """Resamples segments with replacement according to their relative weights.


    Parameters
    ----------
    smallest_allowed_weight, largest_allowed_weight, seed
        See :class:`Resampler` class documentation for details.

    """

    def resample(self, bin, target_count):
        weights = bin.weights
        total_weight = weights.sum()

        counts = self.rng.multinomial(n=target_count, pvals=weights / total_weight)

        new_weight = total_weight / target_count
        wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))
        for segment, count in zip(bin, counts):
            for _ in range(count):
                yield segment.copy(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
