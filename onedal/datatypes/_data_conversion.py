# ==============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from onedal import _default_backend as backend


def _apply_and_pass(func, *args, **kwargs):
    if len(args) == 1:
        return func(args[0], **kwargs)
    return tuple(map(lambda arg: func(arg, **kwargs), args))


def _convert_one_to_table(arg, queue=None):
    # All inputs for table conversion must be array-like or sparse, not scalars
    return backend.to_table(np.atleast_2d(arg) if np.isscalar(arg) else arg, queue)


def to_table(*args, queue=None):
    """Create oneDAL tables from scalars and/or arrays.

    Note: this implementation can be used with scipy.sparse, numpy ndarrays,
    dpctl/dpnp usm_ndarrays, array API standard arrays, and scalars. Tables
    will use pointers to the original array data. Scalars and non-contiguous
    arrays will be copies. Arrays may be modified in-place by oneDAL during
    computation. This works for data located on CPU and SYCL-enabled Intel GPUs.
    Each array may only be of a single datatype (i.e. each must be homogeneous).

    Parameters
    ----------
    *args : scalar, numpy array, sycl_usm_ndarray, array API standard array,
        csr_matrix, or csr_array
        arg1, arg2... The arrays should be given as arguments.

    queue : SyclQueue or None, default=None
        A dpctl or oneDAL backend python representation of a SYCL Queue or None

    Returns
    -------
    tables: oneDAL homogeneous tables
    """
    return _apply_and_pass(_convert_one_to_table, *args, queue=queue)


def from_table(*args, array=None):
    """Create 2 dimensional arrays from oneDAL tables.

    Note: this implementation will convert any table to numpy ndarrays,
    dpctl/dpnp usm_ndarrays, and array API standard arrays of designated
    type. By default, from_table will return numpy arrays and can only
    return other types when necessary object attributes exist (i.e.
    ``__sycl_usm_array_interface__`` or ``__array_namespace__``).

    Parameters
    ----------
    *args : single or multiple python oneDAL tables
        arg1, arg2... The arrays should be given as arguments.

    array : array-like or None, default=None
        python object representing return type. Accessed for conversion
        namespace when sycl_usm_array type or array API standard type

    Returns
    -------
    arrays: numpy arrays, sycl_usm_ndarrays, or array API standard arrays
    """

    func = backend.from_table
    if isintance(array, np.ndarray):
        pass
    elif hasattr(array, "__sycl_usm_array_interface__"):
        # oneDAL returns tables with None sycl queue for CPU sycl queue inputs.
        # This workaround is necessary for the functional preservation
        # of the compute-follows-data execution.
        device = array.sycl_queue if array.sycl_device.is_cpu else None
        if hasattr(array, "__array_namespace__"):
            func = lambda x: array.__array_namespace__().asarray(x, device=device)
        elif hasattr(array, "_create_from_usm_ndarray"):  # signifier of dpnp < 0.19
            xp = array._array_obj.__array_namespace__()
            func = lambda x: array._create_from_usm_ndarray(xp.asarray(x, device=device))
    elif hasattr(array, "__array_namespace__"):
        func = array.__array_namespace__().from_dlpack
    return _apply_and_pass(func, *args)
