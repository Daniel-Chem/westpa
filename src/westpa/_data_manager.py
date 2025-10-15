import logging
from operator import attrgetter

import numpy as np
from h5py import h5s

from .core.data_manager import (
    WESTDataManager,
    seg_index_dtype,
    seg_id_dtype,
    summary_table_dtype,
    normalize_dataset_options,
    require_dataset_from_dsopts,
)
from .core.segment import Segment

logger = logging.getLogger(__name__)


class DataManager(WESTDataManager):

    def __init__(self, h5filename):
        super().__init__()
        self.we_h5filename = h5filename

    # Unlike in WESTDataManager, we don't write progress coordinates until
    # after propagation is complete, and we don't call update_iter_group_links().
    def prepare_iteration(self, n_iter, segments):
        """Prepare for a new iteration by creating space to store the new iteration's data.
        The number of segments, their IDs, and their lineage must be determined and included
        in the set of segments passed in.

        Parameters
        ----------
        n_iter : int
            Iteration number.
        segments : iterable of Segment
            Initial segment data (weights, initial states, and lineage).

        """
        logger.debug('preparing HDF5 group for iteration %d (%d segments)' % (n_iter, len(segments)))

        # Ensure we have a list for guaranteed ordering
        init = n_iter == 0
        segments = list(segments)
        n_particles = len(segments)

        with self.lock:
            if not init:
                # Create a table of summary information about each iteration
                summary_table = self.we_h5file['summary']
                if len(summary_table) < n_iter:
                    summary_table.resize((n_iter + 1,))

            iter_group = self.require_iter_group(n_iter)

            for linkname in ('seg_index', 'wtgraph'):
                try:
                    del iter_group[linkname]
                except KeyError:
                    pass

            # everything indexed by [particle] goes in an index table
            seg_index_table_ds = iter_group.create_dataset('seg_index', shape=(n_particles,), dtype=seg_index_dtype)
            # unfortunately, h5py doesn't like in-place modification of individual fields; it expects
            # tuples. So, construct everything in a numpy array and then dump the whole thing into hdf5
            # In fact, this appears to be an h5py best practice (collect as much in ram as possible and then dump)
            seg_index_table = seg_index_table_ds[...]

            if not init:
                summary_row = np.zeros((1,), dtype=summary_table_dtype)
                summary_row['n_particles'] = n_particles
                summary_row['norm'] = np.add.reduce(list(map(attrgetter('weight'), segments)))
                summary_table[n_iter - 1] = summary_row

            total_parents = 0
            for seg_id, segment in enumerate(segments):
                if segment.seg_id is not None:
                    assert segment.seg_id == seg_id
                else:
                    segment.seg_id = seg_id
                # Parent must be set, though what it means depends on initpoint_type
                assert segment.parent_id is not None
                segment.seg_id = seg_id
                seg_index_table[seg_id]['status'] = segment.status
                seg_index_table[seg_id]['weight'] = segment.weight
                seg_index_table[seg_id]['parent_id'] = segment.parent_id
                seg_index_table[seg_id]['wtg_n_parents'] = len(segment.wtg_parent_ids)
                seg_index_table[seg_id]['wtg_offset'] = total_parents
                total_parents += len(segment.wtg_parent_ids)

            if total_parents > 0:
                wtgraph_ds = iter_group.create_dataset('wtgraph', (total_parents,), seg_id_dtype, compression='gzip', shuffle=True)
                parents = np.empty((total_parents,), seg_id_dtype)

                for seg_id, segment in enumerate(segments):
                    offset = seg_index_table[seg_id]['wtg_offset']
                    extent = seg_index_table[seg_id]['wtg_n_parents']
                    parent_list = list(segment.wtg_parent_ids)
                    parents[offset : offset + extent] = parent_list[:]

                    assert set(parents[offset : offset + extent]) == set(segment.wtg_parent_ids)

                wtgraph_ds[:] = parents

            # Since we accumulated many of these changes in RAM (and not directly in HDF5), propagate
            # the changes out to HDF5
            seg_index_table_ds[:] = seg_index_table

    # When updating segments, we infer pcoord_ndim, pcoord_len, and pcoord_dtype.
    def update_segments(self, n_iter, segments):
        """Update segment information in the HDF5 file; all prior information for each
        segment is overwritten, except for parent and weight transfer information.

        Parameters
        ----------
        n_iter : int
            Iteration number
        segments : iterable of Segment
            Segments with updated information (final state, progress coordinates,
            end-point type, etc.).

        """
        segments = sorted(segments, key=attrgetter('seg_id'))

        # Infer the pcoord shape and dtype from the first segment.
        pcoord_len = segments[0].pcoord.shape[0]
        pcoord_ndim = segments[0].pcoord.shape[1]
        pcoord_dtype = segments[0].pcoord.dtype

        with self.lock:
            summary_table = self.we_h5file['summary']
            n_particles = summary_table[n_iter - 1]['n_particles']
            iter_group = self.get_iter_group(n_iter)

            pcoord_opts = self.dataset_options.get('pcoord', {'name': 'pcoord', 'h5path': 'pcoord', 'compression': False})
            shape = (n_particles, pcoord_len, pcoord_ndim)
            pcoord_ds = require_dataset_from_dsopts(iter_group, pcoord_opts, shape, pcoord_dtype)

            pc_dsid = pcoord_ds.id
            si_dsid = iter_group['seg_index'].id

            seg_ids = [segment.seg_id for segment in segments]
            n_segments = len(segments)
            n_total_segments = si_dsid.shape[0]

            seg_index_entries = np.empty((n_segments,), dtype=seg_index_dtype)
            pcoord_entries = np.empty((n_segments, pcoord_len, pcoord_ndim), dtype=pcoord_dtype)

            pc_msel = h5s.create_simple(pcoord_entries.shape, (h5s.UNLIMITED,) * pcoord_entries.ndim)
            pc_msel.select_all()
            si_msel = h5s.create_simple(seg_index_entries.shape, (h5s.UNLIMITED,))
            si_msel.select_all()
            pc_fsel = pc_dsid.get_space()
            si_fsel = si_dsid.get_space()

            for iseg in range(n_segments):
                seg_id = seg_ids[iseg]
                op = h5s.SELECT_OR if iseg != 0 else h5s.SELECT_SET
                si_fsel.select_hyperslab((seg_id,), (1,), op=op)
                pc_fsel.select_hyperslab((seg_id, 0, 0), (1, pcoord_len, pcoord_ndim), op=op)

            # read summary data so that we have valud parent and weight transfer information
            si_dsid.read(si_msel, si_fsel, seg_index_entries)

            for iseg, (segment, ientry) in enumerate(zip(segments, seg_index_entries)):
                ientry['status'] = segment.status
                ientry['endpoint_type'] = segment.endpoint_type or Segment.SEG_ENDPOINT_UNSET
                ientry['cputime'] = segment.cputime
                ientry['walltime'] = segment.walltime
                ientry['weight'] = segment.weight

                pcoord_entries[iseg] = segment.pcoord

            # write progress coordinates and index using low level HDF5 functions for efficiency
            si_dsid.write(si_msel, si_fsel, seg_index_entries)
            pc_dsid.write(pc_msel, pc_fsel, pcoord_entries)

            # Now, to deal with auxiliary data
            # If any segment has any auxiliary data, then the aux dataset must spring into
            # existence. Each is named according to the name in segment.data, and has shape
            # (n_total_segs, ...) where the ... is the shape of the data in segment.data (and may be empty
            # in the case of scalar data) and dtype is taken from the data type of the data entry
            # compression is on by default for datasets that will be more than 1MiB

            # a mapping of data set name to (per-segment shape, data type) tuples
            dsets = {}

            # First we scan for presence, shape, and data type of auxiliary data sets
            for segment in segments:
                if segment.data:
                    for dsname in segment.data:
                        if dsname.startswith('iterh5/'):
                            continue
                        data = np.asarray(segment.data[dsname], order='C')
                        segment.data[dsname] = data
                        dsets[dsname] = (data.shape, data.dtype)

            # Then we iterate over data sets and store data
            if dsets:
                for dsname, (shape, dtype) in dsets.items():
                    # dset = self._require_aux_dataset(iter_group, dsname, n_total_segments, shape, dtype)
                    try:
                        dsopts = self.dataset_options[dsname]
                    except KeyError:
                        dsopts = normalize_dataset_options({'name': dsname}, path_prefix='auxdata')

                    shape = (n_total_segments,) + shape
                    dset = require_dataset_from_dsopts(
                        iter_group, dsopts, shape, dtype, autocompress_threshold=self.aux_compression_threshold, n_iter=n_iter
                    )
                    if dset is None:
                        # storage is suppressed
                        continue
                    for segment in segments:
                        try:
                            auxdataset = segment.data[dsname]
                        except KeyError:
                            pass
                        else:
                            source_rank = len(auxdataset.shape)
                            source_sel = h5s.create_simple(auxdataset.shape, (h5s.UNLIMITED,) * source_rank)
                            source_sel.select_all()
                            dest_sel = dset.id.get_space()
                            dest_sel.select_hyperslab((segment.seg_id,) + (0,) * source_rank, (1,) + auxdataset.shape)
                            dset.id.write(source_sel, dest_sel, auxdataset)
                    if 'delram' in list(dsopts.keys()):
                        del dsets[dsname]

            self.update_iter_h5file(n_iter, segments)
