import h5py
import numpy as np


seg_id_dtype = np.int64  # Up to 9 quintillion segments per iteration; signed so that initial states can be stored negative
n_iter_dtype = np.uint32  # Up to 4 billion iterations
weight_dtype = np.float64  # about 15 digits of precision in weights
utime_dtype = np.float64  # ("u" for Unix time) Up to ~10^300 cpu-seconds
vstr_dtype = h5py.special_dtype(vlen=str)
h5ref_dtype = h5py.special_dtype(ref=h5py.Reference)
binhash_dtype = np.dtype('|S64')

seg_status_dtype = np.uint8
seg_initpoint_dtype = np.uint8
seg_endpoint_dtype = np.uint8
istate_type_dtype = np.uint8
istate_status_dtype = np.uint8

# TODO: Move remaining dtypes here. Export dtypes to package level.
