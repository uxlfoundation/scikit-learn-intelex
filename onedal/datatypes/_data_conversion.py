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

from types import ModuleType

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

    Parameters
    ----------
    *args : {scalar, numpy array, sycl_usm_ndarray, csr_matrix, or csr_array}
        Arguments to be individually converted to oneDAL tables.

    queue : SyclQueue or None, default=None
        SYCL Queue object to be associated with the oneDAL tables. Default
        value None causes no change in data location or queue.

    Returns
    -------
    tables: {oneDAL homogeneous tables}
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


if backend.is_dpc:

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
                    backend.from_table(table), usm_type="device", sycl_queue=sycl_queue
                )
            else:
                # By default DPNP ndarray created with a copy.
                # TODO:
                # investigate why dpnp.array(table, copy=False) doesn't work.
                # Work around with using dpctl.tensor.asarray.
                if isinstance(xp, ModuleType) and xp.__name__ == "dpnp":
                    return xp.array(xp.dpctl.tensor.asarray(table), copy=False)
                else:
                    return xp.asarray(table)

        return backend.from_table(table)

else:

    def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
        # Currently only `__sycl_usm_array_interface__` protocol used to
        # convert into dpnp/dpctl tensors.
        if sua_iface:
            raise RuntimeError(
                "SYCL usm array conversion from table requires the DPC backend"
            )
        return backend.from_table(table)


def from_table(*args, sycl_queue=None, sua_iface=None, xp=None):
    return _apply_and_pass(
        convert_one_from_table, *args, sycl_queue=sycl_queue, sua_iface=sua_iface, xp=xp
    )
