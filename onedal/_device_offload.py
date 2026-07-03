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

from functools import wraps
from operator import xor

import numpy as np

from .datatypes import dlpack_to_numpy
from .utils import _sycl_queue_manager as QM


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
