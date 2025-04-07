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
from .utils import get_tags


def _get_backend(obj, queue, method_name, *data):
    """This function verifies the hardware conditions, data characteristics, and
    estimator parameters necessary for offloading computation to oneDAL. The status
    of this patching is returned as a PatchingConditionsChain object along with a
    boolean flag determining if to be computed with oneDAL. It is assumed that the
    queue (which determined what hardware to possibly use for oneDAL) has been
    previously and extensively collected (i.e. the data has already been checked)."""

    cpu_device = queue is None or getattr(queue.sycl_device, "is_cpu", True)
    gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)

    if cpu_device:
        patching_status = obj._onedal_cpu_supported(method_name, *data)
        return patching_status.get_status(), patching_status

    if gpu_device:
        patching_status = obj._onedal_gpu_supported(method_name, *data)
        if not patching_status.get_status() and get_config()["allow_fallback_to_host"]:
            patching_status = obj._onedal_cpu_supported(method_name, *data)

        return patching_status.get_status(), patching_status

    raise RuntimeError("Device support is not implemented")


def dispatch(obj, method_name, branches, *args, **kwargs):
    if get_config()["use_raw_input"]:
        return branches["onedal"](obj, *args, **kwargs)

    array_api_offload = (
        "array_api_dispatch" in get_config() and get_config()["array_api_dispatch"]
    )

    onedal_array_api = array_api_offload and get_tags(obj)["onedal_array_api"]
    sklearn_array_api = array_api_offload and get_tags(obj)["array_api_support"]

    # We need to avoid a copy to host here if zero_copy supported
    backend = None
    with QM.manage_global_queue(None, *args) as queue:
        if onedal_array_api:
            backend, patching_status = _get_backend(obj, queue, method_name, *args)
            if backend:
                patching_status.write_log(queue=queue, transferred_to_host=False)
                return branches[backend](obj, *args, **kwargs, queue=queue)
            elif sklearn_array_api:
                patching_status.write_log(transferred_to_host=False)
                return branches[backend](obj, *args, **kwargs)

        # move to host because it is necessary for checking
        # we only guarantee onedal_cpu_supported and onedal_gpu_supported are generalized
        # to non-numpy inputs for zero copy estimators. this will eventually be deprecated
        # when all estimators are zero-copy generalized in onedal_cpu_supported and
        # onedal_gpu_supported.
        has_usm_data_for_args, hostargs = _transfer_to_host(*args)
        has_usm_data_for_kwargs, hostvalues = _transfer_to_host(*kwargs.values())

        hostkwargs = dict(zip(kwargs.keys(), hostvalues))
        has_usm_data = has_usm_data_for_args or has_usm_data_for_kwargs

        if backend is None:
            backend, patching_status = _get_backend(obj, queue, method_name, *hostargs)

        if backend:
            patching_status.write_log(queue=queue, transferred_to_host=False)
            return branches[backend](obj, *hostargs, **hostkwargs, queue=queue)
        else:
            if sklearn_array_api and not has_usm_data:
                # dpnp fallback is not handled properly yet.
                patching_status.write_log(transferred_to_host=False)
                return branches[backend](obj, *args, **kwargs)
            else:
                patching_status.write_log()
                return branches[backend](obj, *hostargs, **hostkwargs)


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
