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

import functools

import pytest

from ...utils._third_party import dpctl_available

if dpctl_available:
    import dpctl
    from dpctl.memory import MemoryUSMDevice, MemoryUSMShared


def get_queues(filter_="cpu,gpu"):
    """Get available dpctl.SycQueues for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    filter_ : str, default="cpu,gpu"
        Configure output list with available dpctl.SycQueues for testing.

    Returns
    -------
    list[dpctl.SycQueue]
        The list of dpctl.SycQueue.

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.
    """
    queues = [None] if "cpu" in filter_ else []

    if dpctl_available:
        if dpctl.has_cpu_devices() and "cpu" in filter_:
            queues.append(pytest.param(dpctl.SyclQueue("cpu"), id="SyclQueue_CPU"))
        if dpctl.has_gpu_devices() and "gpu" in filter_:
            queues.append(pytest.param(dpctl.SyclQueue("gpu"), id="SyclQueue_GPU"))

    return queues


def get_memory_usm():
    if dpctl_available:
        return [MemoryUSMDevice, MemoryUSMShared]
    return []


def is_dpctl_device_available(targets):
    if not isinstance(targets, (list, tuple)):
        raise TypeError("`targets` should be a list or tuple of strings.")
    if dpctl_available:
        for device in targets:
            if device == "cpu" and not dpctl.has_cpu_devices():
                return False
            if device == "gpu" and not dpctl.has_gpu_devices():
                return False
        return True
    return False


def pass_if_not_implemented_for_gpu(reason=""):
    assert reason

    def decorator(test):
        @functools.wraps(test)
        def wrapper(queue, *args, **kwargs):
            if queue is not None and queue.sycl_device.is_gpu:
                with pytest.raises(RuntimeError, match=reason):
                    test(queue, *args, **kwargs)
            else:
                test(queue, *args, **kwargs)

        return wrapper

    return decorator
