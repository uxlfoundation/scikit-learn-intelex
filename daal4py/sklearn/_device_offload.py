# ==============================================================================
# Copyright 2014 Intel Corporation
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

import logging
import sys
from collections.abc import Iterable
from functools import wraps

import numpy as np

from ._config import _get_config

try:
    from dpctl import SyclQueue
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray

    dpctl_available = True
except ImportError:
    dpctl_available = False


oneapi_is_available = "daal4py.oneapi" in sys.modules
if oneapi_is_available:
    from daal4py.oneapi import _get_device_name_sycl_ctxt, _get_sycl_ctxt_params


def _get_device_info():
    if oneapi_is_available:
        return _get_device_name_sycl_ctxt(), _get_sycl_ctxt_params()
    return None, dict()


class DummySyclQueue:
    """This class is designed to act like dpctl.SyclQueue
    to allow device dispatching in scenarios when dpctl is not available"""

    class DummySyclDevice:
        def __init__(self, filter_string):
            self._filter_string = filter_string
            self.is_cpu = "cpu" in filter_string
            self.is_gpu = "gpu" in filter_string
            # TODO: check for possibility of fp64 support
            # on other devices in this dummy class
            self.has_aspect_fp64 = self.is_cpu

            if not (self.is_cpu):
                logging.warning(
                    "Device support is limited. "
                    "Please install dpctl for full experience"
                )

        def get_filter_string(self):
            return self._filter_string

    def __init__(self, filter_string):
        self.sycl_device = self.DummySyclDevice(filter_string)


def _copy_to_usm(queue, array):
    if not dpctl_available:
        raise RuntimeError(
            "dpctl need to be installed to work " "with __sycl_usm_array_interface__"
        )

    if hasattr(array, "__array__"):

        try:
            mem = MemoryUSMDevice(array.nbytes, queue=queue)
            mem.copy_from_host(array.tobytes())
            return usm_ndarray(array.shape, array.dtype, buffer=mem)
        except ValueError as e:
            # ValueError will raise if device does not support the dtype
            # retry with float32 (needed for fp16 and fp64 support issues)
            # try again as float32, if it is a float32 just raise the error.
            if array.dtype == np.float32:
                raise e
            return _copy_to_usm(queue, array.astype(np.float32))
    else:
        if isinstance(array, Iterable):
            array = [_copy_to_usm(queue, i) for i in array]
        return array


def _transfer_to_host(queue, *data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        if usm_iface is not None:
            if not dpctl_available:
                raise RuntimeError(
                    "dpctl need to be installed to work "
                    "with __sycl_usm_array_interface__"
                )
            if queue is not None:
                if queue.sycl_device != usm_iface["syclobj"].sycl_device:
                    raise RuntimeError(
                        "Input data shall be located " "on single target device"
                    )
            else:
                queue = usm_iface["syclobj"]

            buffer = as_usm_memory(item).copy_to_host()
            order = "C"
            if usm_iface["strides"] is not None:
                if usm_iface["strides"][0] < usm_iface["strides"][1]:
                    order = "F"
            item = np.ndarray(
                shape=usm_iface["shape"],
                dtype=usm_iface["typestr"],
                buffer=buffer,
                order=order,
            )
            has_usm_data = True
        else:
            has_host_data = True

        mismatch_host_item = usm_iface is None and item is not None and has_usm_data
        mismatch_usm_item = usm_iface is not None and has_host_data

        if mismatch_host_item or mismatch_usm_item:
            raise RuntimeError("Input data shall be located on single target device")

        host_data.append(item)
    return queue, host_data


def _get_global_queue():
    target = _get_config()["target_offload"]
    d4p_target, _ = _get_device_info()
    if d4p_target == "host":
        d4p_target = "cpu"

    QueueClass = DummySyclQueue if not dpctl_available else SyclQueue

    if target != "auto":
        if d4p_target is not None and d4p_target != target:
            if not isinstance(target, str):
                if d4p_target not in target.sycl_device.get_filter_string():
                    raise RuntimeError(
                        "Cannot use target offload option "
                        "inside daal4py.oneapi.sycl_context"
                    )
            else:
                raise RuntimeError(
                    "Cannot use target offload option "
                    "inside daal4py.oneapi.sycl_context"
                )
        if isinstance(target, QueueClass):
            return target
        return QueueClass(target)
    if d4p_target is not None:
        return QueueClass(d4p_target)
    return None


def _get_host_inputs(*args, **kwargs):
    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return q, hostargs, hostkwargs


def _extract_usm_iface(*args, **kwargs):
    allargs = (*args, *kwargs.values())
    if len(allargs) == 0:
        return None
    return getattr(allargs[0], "__sycl_usm_array_interface__", None)


def _run_on_device(func, queue, obj=None, *args, **kwargs):
    def dispatch_by_obj(obj, func, *args, **kwargs):
        if obj is not None:
            return func(obj, *args, **kwargs)
        return func(*args, **kwargs)

    if queue is not None and oneapi_is_available:
        from daal4py.oneapi import _get_in_sycl_ctxt, sycl_context

        if _get_in_sycl_ctxt() is False:
            host_offload = _get_config()["allow_fallback_to_host"]

            with sycl_context(
                "gpu" if queue.sycl_device.is_gpu else "cpu",
                host_offload_on_fail=host_offload,
            ):
                return dispatch_by_obj(obj, func, *args, **kwargs)
    return dispatch_by_obj(obj, func, *args, **kwargs)


def support_usm_ndarray(freefunc=False):
    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            usm_iface = _extract_usm_iface(*args, **kwargs)
            q, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            result = _run_on_device(func, q, obj, *hostargs, **hostkwargs)
            if usm_iface is not None and hasattr(result, "__array_interface__"):
                return _copy_to_usm(q, result)
            return result

        if freefunc:

            @wraps(func)
            def wrapper_free(*args, **kwargs):
                return wrapper_impl(None, *args, **kwargs)

            return wrapper_free

        @wraps(func)
        def wrapper_with_self(self, *args, **kwargs):
            return wrapper_impl(self, *args, **kwargs)

        return wrapper_with_self

    return decorator
