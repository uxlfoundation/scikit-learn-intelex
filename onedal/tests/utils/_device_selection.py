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
from collections.abc import Iterable

import pytest

from onedal.utils._third_party import SyclQueue


# lru_cache is used to limit the number of SyclQueues generated
@functools.lru_cache()
def get_queues(filter_: str = "cpu,gpu") -> list[SyclQueue]:
    """Get available dpctl.SycQueues for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    filter_ : str, default="cpu,gpu"
        Configure output list with availabe SyclQueues for testing.
        SyclQueues are generated from a comma-separated string with
        each element conforming to SYCL's ``filter_selector``.

    Returns
    -------
    list[SyclQueue]
        The list of SyclQueues.

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


def is_sycl_device_available(targets: Iterable[str]) -> bool:
    """Check if a SYCL device is available.

    This is meant to be used for testing purposes only.
    The check succeeds if all SYCL devices in targets are
    available.

    Parameters
    ----------
    targets : Iterable[str]
        SYCL filter strings of possible devices.

    Returns
    -------
    bool
        Flag if all of the SYCL targets are available.

    """
    if not isinstance(targets, Iterable):
        raise TypeError("`targets` should be an iterable of strings.")
    for device in targets:
        try:
            SyclQueue(device)
        except [RuntimeError, ValueError]:
            return False
    return True


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
