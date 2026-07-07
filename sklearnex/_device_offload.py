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

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Union

from sklearn.utils import get_tags

from daal4py.sklearn._utils import sklearn_check_version
from onedal._device_offload import _get_host_inputs, _transfer_to_host
from onedal.datatypes import copy_to_dpnp
from onedal.utils import _sycl_queue_manager as QM
from onedal.utils._array_api import _asarray, _get_sycl_namespace, _is_numpy_namespace
from onedal.utils._third_party import is_dpnp_ndarray

from ._config import get_config
from ._utils import PatchingConditionsChain
from .base import oneDALEstimator
from .utils._array_api import get_namespace


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
        if not patching_status.get_status() and get_config()["allow_fallback_to_host"]:
            QM.fallback_to_host()
            return None, patching_status
        return patching_status.get_status(), patching_status

    if get_config()["allow_fallback_to_host"]:
        # This may trigger if the ``onedal.utils._sycl_queue_manager.__globals.non_queue``
        # object is the queue (e.g. if non-SYCL device data is encountered)
        QM.fallback_to_host()
        return None, None
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
    by the 'allow_fallback_to_host' config parameter.

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

    # Determine if array_api dispatching is enabled, and if estimator is capable
    onedal_array_api = _array_api_offload() and get_tags(obj).onedal_array_api
    sklearn_array_api = _array_api_offload() and get_tags(obj).array_api_support

    # backend can only be a boolean or None, None signifies an unverified backend
    backend: "bool | None" = None

    # The _sycl_queue_manager verifies all arguments are on a single SYCL device or
    # cpu and will otherwise throw an error. If located on a non-SYCL, non-CPU
    # device, a special queue is set which will cause a failure in ``_get_backend``
    # Comment 2026-04-27: as of scikit-learn1.9, the behavior described above should
    # no longer apply - instead, data should be moved to the namespace and device
    # of either 'X' (in estimators) or 'y' (in metrics) if it wasn't originally there.
    # But note that this requires doing the movements manually in every estimator.
    # It should not move things right here, because up to this point, 'y' is whatever
    # the user supplied, which can be strings for classifiers for example, which cannot
    # be moved to a GPU device.
    context = (
        QM.manage_global_queue(None, *args)
        if not _array_api_offload() or not sklearn_check_version("1.9")
        else QM.manage_global_queue(None, *(args[:1]))
    )
    with context:
        if onedal_array_api:
            backend, patching_status = _get_backend(obj, method_name, *args)
            if backend:
                queue = QM.get_global_queue()
                patching_status.write_log(queue=queue, transferred_to_host=False)
                return branches["onedal"](obj, *args, **kwargs, queue=queue)
            elif sklearn_array_api and backend is False:
                patching_status.write_log(transferred_to_host=False)
                return branches["sklearn"](obj, *args, **kwargs)

        # move data to host because of multiple reasons: array_api fallback to host
        # and non array_api supporting oneDAL code
        _, hostargs = _transfer_to_host(*args)
        _, hostvalues = _transfer_to_host(*kwargs.values())

        hostkwargs = dict(zip(kwargs.keys(), hostvalues))

        while backend is None:
            backend, patching_status = _get_backend(obj, method_name, *hostargs)

        if backend:
            queue = QM.get_global_queue()
            patching_status.write_log(queue=queue, transferred_to_host=False)
            return branches["onedal"](obj, *hostargs, **hostkwargs, queue=queue)
        else:
            if sklearn_array_api:
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
        # In case ARRAY API is enabled the result is already converted to the required type
        if _array_api_offload() and get_tags(self).onedal_array_api:
            # When transform_output is polars/pandas, sklearn's _set_output
            # wrapper calls pl.DataFrame(result) which can't handle GPU arrays.
            # Transfer to host so sklearn can wrap into the requested format.
            if func.__name__ in ("transform", "fit_transform") and (
                get_config().get("transform_output")
                not in (
                    "default",
                    None,
                )
                or getattr(self, "_sklearn_output_config", {}).get("transform", "default")
                != "default"
            ):
                _, (result,) = _transfer_to_host(result)
            return result
        if not (len(args) == 0 and len(kwargs) == 0):
            data = (*args, *kwargs.values())[0]
            # When transform_output is polars/pandas, sklearn's _set_output
            # wrapper calls pl.DataFrame(result) which can't handle GPU arrays.
            # Transfer to host so sklearn can wrap into the requested format.
            if func.__name__ in ("transform", "fit_transform") and (
                get_config().get("transform_output")
                not in (
                    "default",
                    None,
                )
                or getattr(self, "_sklearn_output_config", {}).get("transform", "default")
                != "default"
            ):
                _, (result,) = _transfer_to_host(result)
                return result

            if usm_iface := getattr(data, "__sycl_usm_array_interface__", None):
                queue = usm_iface["syclobj"]
                return copy_to_dpnp(queue, result)

            if get_config().get("transform_output") in ("default", None):
                if hasattr(data, "dtype"):
                    xp, is_array_api = get_namespace(data)
                    if is_array_api and not _is_numpy_namespace(xp):
                        device = getattr(data, "device", None)
                        if isinstance(result, tuple):
                            result = tuple(xp.asarray(r, device=device) for r in result)
                        elif not isinstance(result, (int, float)):
                            result = xp.asarray(result, device=device)
        return result

    return wrapper


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

        if "queue" not in kwargs and "queue" in inspect.signature(func).parameters:
            if usm_iface := getattr(args[0], "__sycl_usm_array_interface__", None):
                kwargs["queue"] = usm_iface["syclobj"]

        if kwargs.get("queue") is not None:
            # Device path — function accepts queue, pass device data directly
            result = invoke_func(self, *args, **kwargs)
        else:
            # Host path — sklearn function or host data, transfer to host
            if len(args) == 0 and len(kwargs) == 0:
                return invoke_func(self, *args, **kwargs)

            with QM.manage_global_queue(None, *args) as queue:
                hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
                result = invoke_func(self, *hostargs, **hostkwargs)
                if queue and hasattr(args[0], "__sycl_usm_array_interface__"):
                    return copy_to_dpnp(queue, result)

        data = (*args, *kwargs.values())[0]
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
