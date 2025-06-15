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

from onedal.utils._third_party import dpctl_available, SyclQueue

# lru_cache is used to limit the number of SyclQueues generated
@functools.lru_cache()
def get_queues(filter_="cpu,gpu": str) -> list[SyclQueue]:
    """Get available dpctl.SycQueues for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    filter_ : str, default="cpu,gpu"
        Configure output list with availabe SycQueues for testing.
        SyclQueues are generated from a comma-separated string with
        each element conforming to SYCL's ``filter_selector``.

    Returns
    -------
    list[SyclQueue]
        The list of SycQueues.

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.
    """
    queues = [None] if "cpu" in filter_ else []

    for i in filter_.split(","):
        try:
            queues.append(pytest.param(SyclQueue(i), id=f"SyclQueue_{i.upper()}"))
        except [RuntimeError, ValueError]:
            pass

    return queues


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
