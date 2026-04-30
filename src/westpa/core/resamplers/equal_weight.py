import abc

import numpy as np

from .base import Resampler


class EqualWeightResampler(Resampler):

    @abc.abstractmethod
    def get_sample_counts(self, normalized_weights, target_count):
        pass

    def resample(self, bin, target_count):
        weights = bin.weights()
        total_weight = weights.sum()

        counts = self.get_sample_counts(weights / total_weight, target_count)

        wtg_parent_ids = set.union(*(segment.wtg_parent_ids for segment in bin))
        new_weight = total_weight / target_count

        new_segments = set()
        for segment, count in zip(bin, counts):
            for _ in range(count):
                new_segment = segment.replace(weight=new_weight, wtg_parent_ids=wtg_parent_ids)
                new_segments.add(new_segment)

        bin.clear()
        bin |= new_segments

        return bin


class MultinomialResampler(EqualWeightResampler):
    """Randomly samples trajectories according to their relative weights.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Pseudo-random number generator to use.

    """

    def get_sample_counts(self, normalized_weights, target_count):
        return self.rng.multinomial(n=target_count, pvals=normalized_weights)


class ResidualResampler(EqualWeightResampler):
    """Implements residual resampling (described `here <https://arxiv.org/abs/1806.00860>`_ in Algorithm 8.1).

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Pseudo-random number generator to use.

    """

    def get_sample_counts(self, normalized_weights, target_count):
        # See Algorithm 8.1 in https://arxiv.org/abs/1806.00860.
        nd = target_count * normalized_weights
        nd_floor = np.floor(nd)
        delta = nd - nd_floor
        trials = round(delta.sum())
        if trials == 0:  # happens when bin.count == 1
            return [target_count]
        else:
            r = self.rng.multinomial(n=trials, pvals=delta / trials)
            return map(round, nd_floor + r)
