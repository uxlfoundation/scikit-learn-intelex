#===============================================================================
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
#===============================================================================

from ._config import get_config
from ._utils import get_patch_message
from functools import wraps


def _get_device_info_from_daal4py():
    import sys
    if 'daal4py.oneapi' in sys.modules:
        from daal4py.oneapi import _get_device_name_sycl_ctxt, _get_sycl_ctxt_params
        return _get_device_name_sycl_ctxt(), _get_sycl_ctxt_params()
    return None, dict()


def _get_global_queue():
    target = get_config()['target_offload']
    d4p_target, _ = _get_device_info_from_daal4py()
    if d4p_target == 'host':
        d4p_target = 'cpu'

    if target != 'auto' or d4p_target is not None:
        try:
            from dpctl import SyclQueue
        except ImportError:
            raise RuntimeError("dpctl need to be installed for device offload")

    if target != 'auto':
        if d4p_target is not None and \
           d4p_target != target and \
           d4p_target not in target.sycl_device.get_filter_string():
            raise RuntimeError("Cannot use target offload option "
                               "inside daal4py.oneapi.sycl_context")
        if isinstance(target, SyclQueue):
            return target
        return SyclQueue(target)
    if d4p_target is not None:
        return SyclQueue(d4p_target)
    return None


def _get_backend(obj, queue, method_name, *data):
    cpu_device = queue is None or queue.sycl_device.is_cpu
    gpu_device = queue is not None and queue.sycl_device.is_gpu

    if (cpu_device and obj._onedal_cpu_supported(method_name, *data)) or \
       (gpu_device and obj._onedal_gpu_supported(method_name, *data)):
        return 'onedal', queue
    if cpu_device:
        return 'sklearn', None

    _, d4p_options = _get_device_info_from_daal4py()
    allow_fallback = get_config()['allow_fallback_to_host'] or \
        d4p_options.get('host_offload_on_fail', False)

    if gpu_device and allow_fallback:
        if obj._onedal_cpu_supported(method_name, *data):
            return 'onedal', None
        return 'sklearn', None

    raise RuntimeError("Device support is not implemented")


def dispatch(obj, method_name, branches, *args, **kwargs):
    import logging
    from daal4py.sklearn._utils import _transfer_to_host

    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))

    backend, q = _get_backend(obj, q, method_name, *hostargs)

    logging.info(f"sklearn.{method_name}: {get_patch_message(backend, q)}")
    if backend == 'onedal':
        return branches[backend](obj, *hostargs, **hostkwargs, queue=q)
    if backend == 'sklearn':
        return branches[backend](obj, *hostargs, **hostkwargs)
    raise RuntimeError(f'Undefined backend {backend} in {method_name}')


def wrap_output_data(func):
    from daal4py.sklearn._utils import _copy_to_usm

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data = (*args, *kwargs.values())
        usm_iface = getattr(data[0], '__sycl_usm_array_interface__', None)
        result = func(self, *args, **kwargs)
        if usm_iface is not None:
            return _copy_to_usm(usm_iface['syclobj'], result)
        return result
    return wrapper
