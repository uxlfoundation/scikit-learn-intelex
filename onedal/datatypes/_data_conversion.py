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

def return_type_constructor(array):
    """generator function for converting oneDAL tables to arrays.

    Note: this implementation will convert any table to numpy ndarrays,
    scipy csr_arrays, dpctl/dpnp usm_ndarrays, and array API standard
    arrays of designated type. By default, from_table will return numpy
    arrays and can only return other types when necessary object
    attributes exist (i.e. ``__sycl_usm_array_interface__`` or
    ``__array_namespace__``).

    Parameters
    ----------
    array : array-like or None
        python object representing an array instance of the return type
        for converting oneDAL tables. Arrays are queried for conversion
        namespace when of sycl_usm_array type or array API standard type.
        When set to None, will return numpy arrays or scipy csr arrays.

    Returns
    -------
    func: callable
        a function which takes in a single table input and returns an array
    """
    func = backend.from_table
    if isinstance(array, np.ndarray) or array is None:
        pass
    elif hasattr(array, "__sycl_usm_array_interface__"):
        # oneDAL returns tables without sycl queues for CPU sycl queue inputs.
        # This workaround is necessary for the functional preservation
        # of the compute-follows-data execution.
        device = array.sycl_queue if array.sycl_device.is_cpu else None
        # Its important to note why the __sycl_usm_array_interface__ is
        # prioritized: it provides finer-grained control of SYCL queues and the 
        # related SYCL devices which are generally unavailable via DLPack
        # representations (such as SYCL contexts, SYCL sub-devices, etc.).
        if hasattr(array, "__array_namespace__"):
            xp = array.__array_namespace__()
            func = lambda x: xp.asarray(x, device=device)
        elif hasattr(array, "_create_from_usm_ndarray"):  # signifier of dpnp < 0.19
            xp = array._array_obj.__array_namespace__()
            from_usm = array._create_from_usm_ndarray
            func = lambda x: from_usm(xp.asarray(x, device=device))
    elif hasattr(array, "__array_namespace__"):
        func = array.__array_namespace__().from_dlpack
    return func

def from_table(*args, like=None):
    """Create 2 dimensional arrays from oneDAL tables.

    Note: this implementation will convert any table to numpy ndarrays,
    scipy csr_arrays, dpctl/dpnp usm_ndarrays, and array API standard
    arrays of designated type. By default, from_table will return numpy
    arrays and can only return other types when necessary object
    attributes exist (i.e. ``__sycl_usm_array_interface__`` or
    ``__array_namespace__``).

    Parameters
    ----------
    *args : single or multiple python oneDAL tables
        arg1, arg2... The arrays should be given as arguments.

    like : callable, array-like or None, default=None
        python object representing an array instance of the return type
        or function capable of converting oneDAL tables into arrays of
        desired type. Arrays are queried for conversion namespace when
        of sycl_usm_array type or array API standard type. When set to
        None, will return numpy arrays or scipy csr arrays.

    Returns
    -------
    arrays: numpy arrays, sycl_usm_ndarrays, or array API standard arrays
    """
    func = like if callable(like) else return_type_constructor(like)
    return _apply_and_pass(func, *args)
