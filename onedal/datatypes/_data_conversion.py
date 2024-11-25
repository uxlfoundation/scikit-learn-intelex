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

import warnings

import numpy as np

from onedal import _backend, _is_dpc_backend


def _apply_and_pass(func, *args, **kwargs):
    if len(args) == 1:
        return func(args[0], **kwargs)
    return tuple(map(lambda arg: func(arg, **kwargs), args))


def _convert_one_to_table(arg):
    # All inputs for table conversion must be array-like or sparse, not scalars
    return _backend.to_table(np.atleast_2d(arg) if np.isscalar(arg) else arg)


def to_table(*args):
    """Create oneDAL tables from scalars and/or arrays.

    Note: this implementation can be used with contiguous scipy.sparse, numpy
    ndarrays, DPCTL/DPNP usm_ndarrays and scalars. Tables will use pointers to the
    original array data. Scalars will be copies. Arrays may be modified in-
    place by oneDAL during computation. This works for data located on CPU and
    SYCL-enabled Intel GPUs. Each array may only be of a single datatype (i.e.
    each must be homogeneous).

    Parameters
    ----------
    *args : {scalar, numpy array, sycl_usm_ndarray, csr_matrix, or csr_array}
        arg1, arg2... The arrays should be given as arguments.

    Returns
    -------
    tables: {oneDAL homogeneous tables}
    """
    return _apply_and_pass(_convert_one_to_table, *args)


if _is_dpc_backend:

    try:
        # try/catch is used here instead of dpep_helpers because
        # of circular import issues of _data_conversion.py and
        # utils/validation.py. This is a temporary fix until the
        # issue with dpnp is addressed, at which point this can
        # be removed entirely.
        import dpnp

        def _table_to_array(table, xp=None):
            # By default DPNP ndarray created with a copy.
            # TODO:
            # investigate why dpnp.array(table, copy=False) doesn't work.
            # Work around with using dpctl.tensor.asarray.
            if xp == dpnp:
                return dpnp.array(dpnp.dpctl.tensor.asarray(table), copy=False)
            else:
                return xp.asarray(table)

    except ImportError:

        def _table_to_array(table, xp=None):
            return xp.asarray(table)

    from ..common._policy import _HostInteropPolicy

    def _convert_to_supported(policy, *data, xp=np):
        def func(x):
            return x

        # CPUs support FP64 by default
        if isinstance(policy, _HostInteropPolicy):
            return _apply_and_pass(func, *data)

        # It can be either SPMD or DPCPP policy
        device = policy._queue.sycl_device

        def convert_or_pass(x):
            if (x is not None) and (x.dtype == xp.float64):
                warnings.warn(
                    "Data will be converted into float32 from "
                    "float64 because device does not support it",
                    RuntimeWarning,
                )
                return xp.astype(x, dtype=xp.float32)
            else:
                return x

        if not device.has_aspect_fp64:
            func = convert_or_pass

        return _apply_and_pass(func, *data)

    def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
        # Currently only `__sycl_usm_array_interface__` protocol used to
        # convert into dpnp/dpctl tensors.
        if sua_iface:
            if (
                sycl_queue
                and sycl_queue.sycl_device.is_cpu
                and table.__sycl_usm_array_interface__["syclobj"] is None
            ):
                # oneDAL returns tables with None sycl queue for CPU sycl queue inputs.
                # This workaround is necessary for the functional preservation
                # of the compute-follows-data execution.
                # Host tables first converted into numpy.narrays and then to array from xp
                # namespace.
                return xp.asarray(
                    _backend.from_table(table), usm_type="device", sycl_queue=sycl_queue
                )
            else:
                return _table_to_array(table, xp=xp)

        return _backend.from_table(table)

else:

    def _convert_to_supported(policy, *data, xp=np):
        def func(x):
            return x

        return _apply_and_pass(func, *data)

    def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
        # Currently only `__sycl_usm_array_interface__` protocol used to
        # convert into dpnp/dpctl tensors.
        if sua_iface:
            raise RuntimeError(
                "SYCL usm array conversion from table requires the DPC backend"
            )
        return _backend.from_table(table)


def from_table(*args, sycl_queue=None, sua_iface=None, xp=None):
    return _apply_and_pass(
        convert_one_from_table, *args, sycl_queue=sycl_queue, sua_iface=sua_iface, xp=xp
    )
