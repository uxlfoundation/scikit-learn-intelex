# ==============================================================================
# Copyright 2023 Intel Corporation
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

import inspect
import logging
from collections.abc import Iterable
from functools import wraps

import numpy as np
from sklearn import get_config

from ._config import _get_config
from .utils import _sycl_queue_manager as QM
from .utils._array_api import _asarray, _is_numpy_namespace
from .utils._dpep_helpers import dpctl_available, dpnp_available

if dpctl_available:
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray
else:
    from onedal import _dpc_backend

    SyclQueue = getattr(_dpc_backend, "SyclQueue", None)

logger = logging.getLogger("sklearnex")


def supports_queue(func):
    """
    Decorator that updates the global queue based on provided queue and global configuration.
    If a `queue` keyword argument is provided in the decorated function, its value will be used globally.
    If no queue is provided, the global queue will be updated from the provided data.
    In either case, all data objects are verified to be on the same device (or on host).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        queue = kwargs.get("queue", None)
        with QM.manage_global_queue(queue, *args) as queue:
            kwargs["queue"] = queue
            result = func(self, *args, **kwargs)
        return result

    return wrapper


if dpnp_available:
    import dpnp

    from .utils._array_api import _convert_to_dpnp


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


def _transfer_to_host(*data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        array_api = getattr(item, "__array_namespace__", lambda: None)()
        if usm_iface is not None:
            if not dpctl_available:
                raise RuntimeError(
                    "dpctl need to be installed to work "
                    "with __sycl_usm_array_interface__"
                )

            buffer = as_usm_memory(item).copy_to_host()
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
            has_usm_data = True
        elif array_api and not _is_numpy_namespace(array_api):
            # `copy`` param for the `asarray`` is not setted.
            # The object is copied only if needed.
            item = np.asarray(item)
            has_host_data = True
        else:
            has_host_data = True

        mismatch_host_item = usm_iface is None and item is not None and has_usm_data
        mismatch_usm_item = usm_iface is not None and has_host_data

        if mismatch_host_item or mismatch_usm_item:
            raise RuntimeError("Input data shall be located on single target device")

        host_data.append(item)
    return has_usm_data, host_data


def _get_host_inputs(*args, **kwargs):
    _, hostargs = _transfer_to_host(*args)
    _, hostvalues = _transfer_to_host(*kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return hostargs, hostkwargs


def support_input_format(func):
    """
    Converts and moves the output arrays of the decorated function
    to match the input array type and device.
    Puts SYCLQueue from data to decorated function arguments.
    """

    def invoke_func(self_or_None, *args, **kwargs):
        if self_or_None is None:
            return func(*args, **kwargs)
        else:
            return func(self_or_None, *args, **kwargs)

    @wraps(func)
    def wrapper_impl(*args, **kwargs):
        # remove self from args if it is a class method
        if inspect.isfunction(func) and "." in func.__qualname__:
            self = args[0]
            args = args[1:]
        else:
            self = None

        # KNeighbors*.fit can not be used with raw inputs, ignore `use_raw_input=True`
        override_raw_input = (
            self
            and self.__class__.__name__ in ("KNeighborsClassifier", "KNeighborsRegressor")
            and func.__name__ == "fit"
        )
        if override_raw_input:
            pretty_name = f"{self.__class__.__name__}.{func.__name__}"
            logger.warning(
                f"Using raw inputs is not supported for {pretty_name}. Ignoring `use_raw_input=True` setting."
            )
        if _get_config()["use_raw_input"] is True and not override_raw_input:
            if "queue" not in kwargs:
                usm_iface = getattr(args[0], "__sycl_usm_array_interface__", None)
                data_queue = usm_iface["syclobj"] if usm_iface is not None else None
                kwargs["queue"] = data_queue
            return invoke_func(self, *args, **kwargs)
        elif len(args) == 0 and len(kwargs) == 0:
            # no arguments, there's nothing we can deduce from them -> just call the function
            return invoke_func(self, *args, **kwargs)

        data = (*args, *kwargs.values())
        # get and set the global queue from the kwarg or data
        with QM.manage_global_queue(kwargs.get("queue"), *args) as queue:
            hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            if "queue" in inspect.signature(func).parameters:
                # set the queue if it's expected by func
                hostkwargs["queue"] = queue
            result = invoke_func(self, *hostargs, **hostkwargs)

            usm_iface = getattr(data[0], "__sycl_usm_array_interface__", None)
            if queue is not None and usm_iface is not None:
                result = _copy_to_usm(queue, result)
                if dpnp_available and isinstance(data[0], dpnp.ndarray):
                    result = _convert_to_dpnp(result)
                return result

        if not get_config().get("transform_output"):
            input_array_api = getattr(data[0], "__array_namespace__", lambda: None)()
            if input_array_api:
                input_array_api_device = data[0].device
                result = _asarray(result, input_array_api, device=input_array_api_device)
        return result

    return wrapper_impl
