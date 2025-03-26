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

from functools import wraps

from onedal._device_offload import _copy_to_usm, _transfer_to_host
from onedal.utils import _sycl_queue_manager as QM
from onedal.utils._array_api import _asarray
from onedal.utils._dpep_helpers import dpnp_available

if dpnp_available:
    import dpnp
    from onedal.utils._array_api import _convert_to_dpnp

from ._config import get_config


def _get_backend(obj, queue, method_name, *data):
    with QM.manage_global_queue(queue, *data) as queue:
        cpu_device = queue is None or getattr(queue.sycl_device, "is_cpu", True)
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)

        if cpu_device:
            patching_status = obj._onedal_cpu_supported(method_name, *data)
            if patching_status.get_status():
                return "onedal", patching_status
            else:
                return "sklearn", patching_status

        allow_fallback_to_host = get_config()["allow_fallback_to_host"]

        if gpu_device:
            patching_status = obj._onedal_gpu_supported(method_name, *data)
            if patching_status.get_status():
                return "onedal", patching_status
            else:
                QM.remove_global_queue()
                if allow_fallback_to_host:
                    patching_status = obj._onedal_cpu_supported(method_name, *data)
                    if patching_status.get_status():
                        return "onedal", patching_status
                    else:
                        return "sklearn", patching_status
                else:
                    return "sklearn", patching_status

    raise RuntimeError("Device support is not implemented")


def dispatch(obj, method_name, branches, *args, **kwargs):
    if get_config()["use_raw_input"] is False:
        with QM.manage_global_queue(None, *args) as queue:
            has_usm_data_for_args, hostargs = _transfer_to_host(*args)
            has_usm_data_for_kwargs, hostvalues = _transfer_to_host(*kwargs.values())
            hostkwargs = dict(zip(kwargs.keys(), hostvalues))

            backend, patching_status = _get_backend(obj, queue, method_name, *hostargs)
            has_usm_data = has_usm_data_for_args or has_usm_data_for_kwargs
            if backend == "onedal":
                # Host args only used before onedal backend call.
                # Device will be offloaded when onedal backend will be called.
                patching_status.write_log(queue=queue, transferred_to_host=False)
                return branches[backend](obj, *hostargs, **hostkwargs, queue=queue)
            if backend == "sklearn":
                if (
                    "array_api_dispatch" in get_config()
                    and get_config()["array_api_dispatch"]
                    and "array_api_support" in obj._get_tags()
                    and obj._get_tags()["array_api_support"]
                    and not has_usm_data
                ):
                    # USM ndarrays are also excluded for the fallback Array API. Currently, DPNP.ndarray is
                    # not compliant with the Array API standard, and DPCTL usm_ndarray Array API is compliant,
                    # except for the linalg module. There is no guarantee that stock scikit-learn will
                    # work with such input data. The condition will be updated after DPNP.ndarray and
                    # DPCTL usm_ndarray enabling for conformance testing and these arrays supportance
                    # of the fallback cases.
                    # If `array_api_dispatch` enabled and array api is supported for the stock scikit-learn,
                    # then raw inputs are used for the fallback.
                    patching_status.write_log(transferred_to_host=False)
                    return branches[backend](obj, *args, **kwargs)
                else:
                    patching_status.write_log()
                    return branches[backend](obj, *hostargs, **hostkwargs)
        raise RuntimeError(
            f"Undefined backend {backend} in " f"{obj.__class__.__name__}.{method_name}"
        )
    else:
        return branches["onedal"](obj, *args, **kwargs)


def wrap_output_data(func):
    """
    Converts and moves the output arrays of the decorated function
    to match the input array type and device.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if not (len(args) == 0 and len(kwargs) == 0):
            data = (*args, *kwargs.values())
            usm_iface = getattr(data[0], "__sycl_usm_array_interface__", None)
            if usm_iface is not None:
                result = _copy_to_usm(usm_iface["syclobj"], result)
                if dpnp_available and isinstance(data[0], dpnp.ndarray):
                    result = _convert_to_dpnp(result)
                return result
            config = get_config()
            if not ("transform_output" in config and config["transform_output"]):
                input_array_api = getattr(data[0], "__array_namespace__", lambda: None)()
                if input_array_api:
                    input_array_api_device = data[0].device
                    result = _asarray(
                        result, input_array_api, device=input_array_api_device
                    )
        return result

    return wrapper
