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

from collections.abc import Callable
from functools import wraps
from typing import Any, Union

from onedal._device_offload import _transfer_to_host
from onedal.datatypes import copy_to_dpnp, copy_to_usm
from onedal.utils import _sycl_queue_manager as QM
from onedal.utils._array_api import _asarray, _is_numpy_namespace
from onedal.utils._third_party import is_dpnp_ndarray

from ._config import config_context, get_config, set_config
from ._utils import PatchingConditionsChain, get_tags
from .base import oneDALEstimator


def _get_backend(
    obj: type[oneDALEstimator], method_name: str, *data
) -> tuple[Union[bool, None], PatchingConditionsChain]:
    """This function verifies the hardware conditions, data characteristics, and
    estimator parameters necessary for offloading computation to oneDAL. The status
    of this patching is returned as a PatchingConditionsChain object along with a
    boolean flag signaling whether the computation can be offloaded to oneDAL or not.
    It is assumed that the queue (which determined what hardware to possibly use for
    oneDAL) has been previously and extensively collected (i.e. the data has already
    been checked using onedal's SyclQueueManager for queues)."""
    queue = QM.get_global_queue()
    cpu_device = queue is None or getattr(queue.sycl_device, "is_cpu", True)
    gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)

    if cpu_device:
        patching_status = obj._onedal_cpu_supported(method_name, *data)
        return patching_status.get_status(), patching_status

    if gpu_device:
        patching_status = obj._onedal_gpu_supported(method_name, *data)
        if (
            not patching_status.get_status()
            and (config := get_config())["allow_fallback_to_host"]
        ):
            QM.fallback_to_host()
            return None, patching_status
        return patching_status.get_status(), patching_status

    raise RuntimeError("Device support is not implemented for the supplied data type.")


if "array_api_dispatch" in get_config():
    _array_api_offload = lambda: get_config()["array_api_dispatch"]
else:
    _array_api_offload = lambda: False


def dispatch(
    obj: type[oneDALEstimator],
    method_name: str,
    branches: dict[Callable, Callable],
    *args,
    **kwargs,
) -> Any:
    """Dispatch object method call to oneDAL if conditionally possible.

    Depending on support conditions, oneDAL will be called, otherwise it will
    fall back to calling scikit-learn.  Dispatching to oneDAL can be influenced
    by the 'use_raw_input' or 'allow_fallback_to_host' config parameters.

    Parameters
    ----------
    obj : object
        Sklearnex object which inherits from oneDALEstimator and contains
        ``onedal_cpu_supported`` and ``onedal_gpu_supported`` methods which
        evaluate oneDAL support.

    method_name : str
        Name of method to be evaluated for oneDAL support.

    branches : dict
        Dictionary containing functions to be called. Only keys 'sklearn' and
        'onedal' are used which should contain the relevant scikit-learn and
        onedal object methods respectively. All functions should accept the
        inputs from *args and **kwargs. Additionally, the onedal object method
        must accept a 'queue' keyword.

    *args : tuple
        Arguments to be supplied to the dispatched method.

    **kwargs : dict
        Keyword arguments to be supplied to the dispatched method.

    Returns
    -------
    unknown : object
        Returned object dependent on the supplied branches. Implicitly the returned
        object types should match for the sklearn and onedal object methods.
    """

    if get_config()["use_raw_input"]:
        return branches["onedal"](obj, *args, **kwargs)

    # Determine if array_api dispatching is enabled, and if estimator is capable
    onedal_array_api = _array_api_offload() and get_tags(obj).onedal_array_api
    sklearn_array_api = _array_api_offload() and get_tags(obj).array_api_support

    # backend can only be a boolean or None, None signifies an unverified backend
    backend: "bool | None" = None

    # config context needs to be saved, as the sycl_queue_manager interacts with
    # target_offload, which can regenerate a GPU queue later on. Therefore if a
    # fallback occurs, then the state of target_offload must be set to default
    # so that later use of get_global_queue only sends to host. We must modify
    # the target offload settings, but we must also set the original value at the
    # end, hence the need of a contextmanager.
    with QM.manage_global_queue(None, *args):
        if onedal_array_api:
            backend, patching_status = _get_backend(obj, method_name, *args)
            if backend:
                queue = QM.get_global_queue()
                patching_status.write_log(queue=queue, transferred_to_host=False)
                return branches["onedal"](obj, *args, **kwargs, queue=queue)
            elif sklearn_array_api and backend is False:
                patching_status.write_log(transferred_to_host=False)
                return branches["sklearn"](obj, *args, **kwargs)

        # move data to host because of multiple reasons: array_api fallback to host,
        # non array_api supporting oneDAL code, issues with usm support in sklearn.
        has_usm_data_for_args, hostargs = _transfer_to_host(*args)
        has_usm_data_for_kwargs, hostvalues = _transfer_to_host(*kwargs.values())

        hostkwargs = dict(zip(kwargs.keys(), hostvalues))
        has_usm_data = has_usm_data_for_args or has_usm_data_for_kwargs

        while backend is None:
            backend, patching_status = _get_backend(obj, method_name, *hostargs)

        if backend:
            queue = QM.get_global_queue()
            patching_status.write_log(queue=queue, transferred_to_host=False)
            return branches["onedal"](obj, *hostargs, **hostkwargs, queue=queue)
        else:
            if sklearn_array_api and not has_usm_data:
                # dpnp fallback is not handled properly yet.
                patching_status.write_log(transferred_to_host=False)
                return branches["sklearn"](obj, *args, **kwargs)
            else:
                patching_status.write_log()
                return branches["sklearn"](obj, *hostargs, **hostkwargs)


def wrap_output_data(func: Callable) -> Callable:
    """Transform function output to match input format.

    Converts and moves the output arrays of the decorated function
    to match the input array type and device.

    Parameters
    ----------
    func : callable
       Function or method which has array data as input.

    Returns
    -------
    wrapper : callable
        Wrapped function or method which will return matching format.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        result = func(self, *args, **kwargs)
        if not (len(args) == 0 and len(kwargs) == 0):
            data = (*args, *kwargs.values())[0]

            if usm_iface := getattr(data, "__sycl_usm_array_interface__", None):
                queue = usm_iface["syclobj"]
                return (
                    copy_to_dpnp(queue, result)
                    if is_dpnp_ndarray(data)
                    else copy_to_usm(queue, result)
                )

            if get_config().get("transform_output") in ("default", None):
                input_array_api = getattr(data, "__array_namespace__", lambda: None)()
                if input_array_api and not _is_numpy_namespace(input_array_api):
                    input_array_api_device = data.device
                    result = _asarray(
                        result, input_array_api, device=input_array_api_device
                    )
        return result

    return wrapper
