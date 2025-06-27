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
import scipy.sparse as sp

from onedal import _default_backend as backend

from ..utils._third_party import is_dpctl_tensor, is_dpnp_ndarray, lazy_import


def _apply_and_pass(func, *args, **kwargs):
    if len(args) == 1:
        return func(args[0], **kwargs)
    return tuple(map(lambda arg: func(arg, **kwargs), args))


def _convert_one_to_table(arg, queue=None):
    # All inputs for table conversion must be array-like or sparse, not scalars
    return backend.to_table(np.atleast_2d(arg) if np.isscalar(arg) else arg, queue)


def to_table(*args, queue=None):
    """Create oneDAL tables from scalars and/or arrays.

    Parameters
    ----------
    *args : scalar, numpy array, sycl_usm_ndarray, csr_matrix, or csr_array
        Arguments to be individually converted to oneDAL tables.

    queue : SyclQueue or None, default=None
        SYCL Queue object to be associated with the oneDAL tables. Default
        value None causes no change in data location or queue.

    queue : SyclQueue or None, default=None
        A dpctl or oneDAL backend python representation of a SYCL Queue or None.

    Returns
    -------
    tables: oneDAL homogeneous tables
        Converted oneDAL tables.

    Notes
    -----
        Tables will use pointers to the original array data. Scalars
        and non-contiguous arrays will be copies. Arrays may be
        modified in-place by oneDAL during computation. Transformation
        is possible only for data located on CPU and SYCL-enabled Intel
        GPUs. Each array may only be of a single data type (i.e. each
        must be homogeneous).
    """
    return _apply_and_pass(_convert_one_to_table, *args, queue=queue)


@lazy_import("array_api_compat")
def _compat_convert(array_api_compat, array):
    return array_api_compat.get_namespace(array).from_dlpack


def return_type_constructor(array):
    """Generate a function for converting oneDAL tables to arrays.

    Parameters
    ----------
    array : array-like or None
        Python object representing an array instance of the return type
        for converting oneDAL tables. Arrays are queried for conversion
        namespace when of sycl_usm_array type or array API standard type.
        When set to None, will return numpy arrays or scipy csr arrays.

    Returns
    -------
    func : callable
        A function which takes in a single table input and returns an array.
    """
    if isinstance(array, np.ndarray) or array is None or sp.issparse(array):
        func = backend.from_table
    elif hasattr(array, "__sycl_usm_array_interface__"):
        # oneDAL returns tables without sycl queues for CPU sycl queue inputs.
        # This workaround is necessary for the functional preservation
        # of the compute-follows-data execution.
        device = array.sycl_queue
        # Its important to note why the __sycl_usm_array_interface__ is
        # prioritized: it provides finer-grained control of SYCL queues and the
        # related SYCL devices which are generally unavailable via DLPack
        # representations (such as SYCL contexts, SYCL sub-devices, etc.).
        if is_dpctl_tensor(array):
            xp = array.__array_namespace__()
            func = lambda x: (
                xp.asarray(x)
                if hasattr(x, "__sycl_usm_array_interface__")
                else xp.asarray(backend.from_table(x), device=device)
            )
        elif is_dpnp_ndarray(array):
            xp = array._array_obj.__array_namespace__()
            from_usm = array._create_from_usm_ndarray
            func = lambda x: from_usm(
                xp.asarray(x)
                if hasattr(x, "__sycl_usm_array_interface__")
                else xp.asarray(backend.from_table(x), device=device)
            )
    elif hasattr(array, "__array_namespace__"):
        func = array.__array_namespace__().from_dlpack
    else:
        try:
            func = _compat_convert(array)
        except ImportError:
            raise TypeError(
                "array type is unsupported, but may be made compatible by installing `array_api_compat`"
            ) from None
    return func


def from_table(*args, like=None):
    """Create 2 dimensional arrays from oneDAL tables.

    oneDAL tables are converted to numpy ndarrays, dpctl tensors, dpnp
    ndarrays, or array API standard arrays of designated type.

    Parameters
    ----------
    *args : single or multiple python oneDAL tables
        The tables should given as individual arguments.

    like : callable, array-like or None, default=None
        Python object representing an array instance of the return type
        or function capable of converting oneDAL tables into arrays of
        desired type. Arrays are queried for conversion namespace when
        of sycl_usm_array type or array API standard type. When set to
        None, will return numpy arrays or scipy csr arrays.

    Returns
    -------
    arrays : numpy arrays, sycl_usm_ndarrays, or array API standard arrays
        Array or tuple of arrays generated from the input oneDAL tables.

    Notes
    -----
    Support for other array types via array_api_compat is possible (e.g.
    PyTorch), but requires its installation specifically, as it is imported
    only when necessary.
    """
    func = like if callable(like) else return_type_constructor(like)
    return _apply_and_pass(func, *args)
