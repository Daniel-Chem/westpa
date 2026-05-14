import logging

import numpy as np
from scipy.spatial.distance import cdist

from .assign import BinMapper, VoronoiBinMapper

logger = logging.getLogger(__name__)


class AdaptiveVoronoiBinMapper(BinMapper):
    """Adaptively place Voronoi sites using the procedure described in Zhang, Jasnow, and Zuckerman (2010). [1]_

    Parameters
    ----------
    nbins : int, default 10
        Number of Voronoi sites.
    metric : str or callable, default 'euclidean'
        Distance metric to use. For available metrics, see the
        ``scipy.spatial.distance.cdist`` documentation.
    metric_kwargs : dict, optional
        Extra arguments to `metric`. See the
        ``scipy.spatial.distance.cdist`` documentation for details.
    update_interval : int, default 1
        Number of iterations between bin updates.
    rng : numpy.random.Generator, optional
        Pseudo-random number generator to use.

    References
    ----------
    .. [1] B.W. Zhang, D. Jasnow, D.M. Zuckerman
       Journal of Chemical Physics, Volume 132, 2010, Page 054107,
       https://doi.org/10.1063/1.3306345.

    """

    def __init__(
        self,
        nbins=10,
        metric='euclidean',
        metric_kwargs=None,
        update_interval=1,
        rng=None,
    ):
        super().__init__()
        self.nbins = nbins
        self.metric = metric
        self.metric_kwargs = metric_kwargs or {}
        self.update_interval = update_interval
        self.rng = np.random.default_rng(rng)

        self.labels = [f'cell_{i}' for i in range(nbins)]

        self.current_mapper = None
        self.last_update = None

    def dfunc(self, x, ys):
        ds = cdist(np.array([x]), ys, metric=self.metric, **self.metric_kwargs)
        return np.array(ds[0], dtype=np.float32)  # type must match coord_t in _assign.pyx

    @property
    def centers(self):
        return self.current_mapper.centers if self.current_mapper else None

    def update_centers(self, coords):
        logger.info('Updating Voronoi centers')

        if len(coords) <= self.nbins:
            self.current_mapper = VoronoiBinMapper(self.dfunc, coords)
            return

        centers = []

        # randomly choose the first center
        idx = self.rng.choice(len(coords))
        centers.append(coords[idx])

        # d_min := distance of each point to nearest center
        d_min = self.dfunc(centers[-1], coords)

        # iteratively add the point with the maximum d_min value
        while len(centers) < self.nbins:
            idx = np.argmax(d_min)
            centers.append(coords[idx])
            d_min = np.minimum(d_min, self.dfunc(centers[-1], coords))

        self.current_mapper = VoronoiBinMapper(self.dfunc, centers)

    def map(self, segments, bins):
        coords = np.array([segment.pcoord[-1] for segment in segments])
        n_iter = segments[0].n_iter

        if self.last_update is None or n_iter - self.last_update == self.update_interval:
            self.update_centers(coords)
            self.last_update = n_iter
        elif n_iter <= self.last_update:
            raise ValueError(f"'n_iter' must be greater than 'last_update' ({n_iter} <= {self.last_update})")

        for i, segment in zip(self.current_mapper.assign(coords), segments):
            bins[i].add(segment)
