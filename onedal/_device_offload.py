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
from functools import wraps
from operator import xor

import numpy as np
from sklearn import get_config

from ._config import _get_config
from .datatypes import copy_to_dpnp, copy_to_usm, dlpack_to_numpy
from .utils import _sycl_queue_manager as QM
from .utils._array_api import _asarray, _get_sycl_namespace, _is_numpy_namespace
from .utils._third_party import is_dpnp_ndarray

logger = logging.getLogger("sklearnex")


def supports_queue(func):
    """Decorator that updates the global queue before function evaluation.

    The global queue is updated based on provided queue and global configuration.
    If a ``queue`` keyword argument is provided in the decorated function, its
    value will be used globally. If no queue is provided, the global queue will
    be updated from the provided data. In either case, all data objects are
    verified to be on the same device (or on host).

    Parameters
    ----------
        func : callable
            Function to be wrapped for SYCL queue use in oneDAL.

    Returns
    -------
        wrapper : callable
            Wrapped function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        queue = kwargs.get("queue", None)
        with QM.manage_global_queue(queue, *args) as queue:
            kwargs["queue"] = queue
            result = func(self, *args, **kwargs)
        return result

    return wrapper


def _transfer_to_host(*data):
    has_usm_data = None

    host_data = []
    for item in data:
        if item is None:
            host_data.append(item)
            continue

        if usm_iface := hasattr(item, "__sycl_usm_array_interface__"):
            xp = item.__array_namespace__()
            item = xp.asnumpy(item)
            has_usm_data = has_usm_data or has_usm_data is None
        elif not isinstance(item, np.ndarray) and (hasattr(item, "__dlpack_device__")):
            item = dlpack_to_numpy(item)

        # set has_usm_data to boolean and use xor to see if they don't match
        if xor((has_usm_data := bool(has_usm_data)), usm_iface):
            raise RuntimeError("Input data shall be located on single target device")

        host_data.append(item)
    return has_usm_data, host_data


def _get_host_inputs(*args, **kwargs):
    _, hostargs = _transfer_to_host(*args)
    _, hostvalues = _transfer_to_host(*kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return hostargs, hostkwargs


def support_input_format(func):
    """Transform input and output function arrays to/from host.

    Converts and moves the output arrays of the decorated function
    to match the input array type and device.
    Puts SYCLQueue from data to decorated function arguments.

    Parameters
    ----------
    func : callable
       Function or method which has array data as input.

    Returns
    -------
    wrapper_impl : callable
        Wrapped function or method which will return matching format.
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
            and _get_config()["use_raw_input"] is True
        )
        if override_raw_input:
            pretty_name = f"{self.__class__.__name__}.{func.__name__}"
            logger.warning(
                f"Using raw inputs is not supported for {pretty_name}. Ignoring `use_raw_input=True` setting."
            )
        if _get_config()["use_raw_input"] is True and not override_raw_input:
            if "queue" not in kwargs:
                if usm_iface := getattr(args[0], "__sycl_usm_array_interface__", None):
                    kwargs["queue"] = usm_iface["syclobj"]
                else:
                    kwargs["queue"] = None
            return invoke_func(self, *args, **kwargs)
        elif len(args) == 0 and len(kwargs) == 0:
            # no arguments, there's nothing we can deduce from them -> just call the function
            return invoke_func(self, *args, **kwargs)

        data = (*args, *kwargs.values())[0]
        # get and set the global queue from the kwarg or data
        with QM.manage_global_queue(kwargs.get("queue"), *args) as queue:
            hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            if "queue" in inspect.signature(func).parameters:
                # set the queue if it's expected by func
                hostkwargs["queue"] = queue
            result = invoke_func(self, *hostargs, **hostkwargs)

            if queue and hasattr(data, "__sycl_usm_array_interface__"):
                return (
                    copy_to_dpnp(queue, result)
                    if is_dpnp_ndarray(data)
                    else copy_to_usm(queue, result)
                )

        if get_config().get("transform_output") in ("default", None):
            input_array_api = getattr(data, "__array_namespace__", lambda: None)()
            if input_array_api and not _is_numpy_namespace(input_array_api):
                input_array_api_device = data.device
                result = _asarray(result, input_array_api, device=input_array_api_device)
        return result

    return wrapper_impl


def support_sycl_format(func):
    # This wrapper enables scikit-learn functions and methods to work with
    # all sycl data frameworks as they no longer support numpy implicit
    # conversion and must be manually converted. This is only necessary
    # when array API is supported but not active.

    @wraps(func)
    def wrapper(*args, **kwargs):
        if (
            not get_config().get("array_api_dispatch", False)
            and _get_sycl_namespace(*args)[2]
        ):
            with QM.manage_global_queue(kwargs.get("queue"), *args):
                if inspect.isfunction(func) and "." in func.__qualname__:
                    self, (args, kwargs) = args[0], _get_host_inputs(*args[1:], **kwargs)
                    return func(self, *args, **kwargs)
                else:
                    args, kwargs = _get_host_inputs(*args, **kwargs)
                    return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper
