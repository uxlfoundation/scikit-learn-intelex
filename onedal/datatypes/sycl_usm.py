# ==============================================================================
# Copyright 2021 Contributors to the oneDAL Project
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

from collections.abc import Iterable

import numpy as np
from ..utils._third_party import lazy_import
     
@lazy_import("dpctl.memory")
@lazy_import("dpctl.tensor")
def _array_to_usm(memorymod, tensormod, queue, array):
    try:
        mem = memorymod.MemoryUSMDevice(array.nbytes, queue=queue)
        mem.copy_from_host(array.tobytes())
        return tensormod.usm_ndarray(array.shape, array.dtype, buffer=mem)
    except ValueError as e:
        # ValueError will raise if device does not support the dtype
        # retry with float32 (needed for fp16 and fp64 support issues)
        # try again as float32, if it is a float32 just raise the error.
        if array.dtype == np.float32:
            raise e
        return _array_to_usm(queue, array.astype(np.float32))

@lazy_import("dpnp")
@lazy_import("dpctl.tensor")
def to_dpnp(dpnpmod, tensormod, array):
    if isinstance(array, tensormod.usm_ndarray):
        return dpnpmod.array(array, copy=False)
    else:
        return array



def copy_to_usm(queue, array):
    if hasattr(array, "__array__"):
        return _array_to_usm(queue, array)
    else:
        if isinstance(array, Iterable):
            array = [copy_to_usm(queue, i) for i in array]
        return array


def copy_to_dpnp(queue, array):
    if hasattr(array, "__array__"):
        return to_dpnp(_array_to_usm(queue, array))
    else:
        if isinstance(array, Iterable):
            array = [copy_to_dpnp(queue, i) for i in array]
        return array


@lazy_import("dpctl.memory")
def _usm_to_array(memorymod, usm_iface, item):
    buffer = memorymod.as_usm_memory(item).copy_to_host()
    order = "C"
    if usm_iface["strides"] is not None and len(usm_iface["strides"]) > 1:
        if usm_iface["strides"][0] < usm_iface["strides"][1]:
            order = "F"
    item = np.ndarray(
        shape=usm_iface["shape"],
        dtype=usm_iface["typestr"],
        buffer=buffer,
        order=order,
    )
    return item






def _transfer_to_host(*data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        if usm_iface:
            item = _convert_usm_to_array(usm_iface, item)
            has_usm_data = True
        elif not isinstance(item, np.ndarray) and (
            device := getattr(item, "__dlpack_device__", None)
        ):
            # check dlpack data location.
            if device() != cpu_dlpack_device:
                if hasattr(item, "to_device"):
                    # use of the "cpu" string as device not officially part of
                    # the array api standard but widely supported
                    item = item.to_device("cpu")
                elif hasattr(item, "to"):
                    # pytorch-specific fix as it is not array api compliant
                    item = item.to("cpu")
                else:
                    raise TypeError(f"cannot move {type(item)} to cpu")

            # convert to numpy
            if hasattr(item, "__array__"):
                # `copy`` param for the `asarray`` is not set.
                # The object is copied only if needed
                item = np.asarray(item)
            else:
                # requires numpy 1.23
                item = np.from_dlpack(item)
            has_host_data = True
        else:
            has_host_data = True

        mismatch_host_item = usm_iface is None and item is not None and has_usm_data
        mismatch_usm_item = usm_iface is not None and has_host_data

        if mismatch_host_item or mismatch_usm_item:
            raise RuntimeError("Input data shall be located on single target device")

        host_data.append(item)
    return has_usm_data, host_data