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
import warnings
from collections.abc import Iterable

import numpy as np
import pytest

from onedal import _dpc_backend
from onedal.utils._third_party import SyclQueue, dpctl_available

if dpctl_available:
    import dpctl

    queue_creation_err = dpctl._sycl_queue.SyclQueueCreationError
else:
    queue_creation_err = (RuntimeError, ValueError)


# SYCL device aspects gating non-universal floating-point dtypes. Anything not
# listed here (fp32, integer types) is assumed supported on every device.
_DTYPE_DEVICE_ASPECTS = {
    "float16": "has_aspect_fp16",
    "float64": "has_aspect_fp64",
}


def _queue_supports_dtype(queue, dtype) -> bool:
    """Return True if ``queue`` (or host, if None) natively supports ``dtype``."""
    if queue is None:
        return True
    aspect = _DTYPE_DEVICE_ASPECTS.get(np.dtype(dtype).name)
    if aspect is None:
        return True
    return bool(getattr(queue.sycl_device, aspect))


# lru_cache is used to limit the number of SyclQueues generated
# @functools.lru_cache()
def get_queues(filter_: str = "cpu,gpu", dtypes=None):
    """Get available dpctl.SycQueues for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    filter_ : str, default="cpu,gpu"
        Configure output list with available SyclQueues for testing.
        SyclQueues are generated from a comma-separated string with
        each element conforming to SYCL's ``filter_selector``.
    dtypes : iterable of numpy dtypes, default=None
        If provided, the returned params include a ``dtype`` dimension
        alongside ``queue``. Each ``(queue, dtype)`` combo is filtered
        by the device's SYCL aspects (``has_aspect_fp16`` /
        ``has_aspect_fp64``) so fp32-only GPUs never receive fp64
        parameters. Host entries (``queue is None``) yield every
        requested dtype.

    Returns
    -------
    list
        When ``dtypes is None`` — legacy behavior: a list of
        ``pytest.param(queue)`` entries (plus a bare ``None`` for the
        CPU host when requested).
        When ``dtypes`` is provided — a list of
        ``pytest.param(queue, dtype)`` entries, with combos unsupported
        by the device filtered out.

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.
    """
    raw_queues = []
    if "cpu" in filter_:
        raw_queues.append((None, None))
    if _dpc_backend is None:
        if "gpu" in filter_:
            warnings.warn(
                "Attempting to get a GPU queue, but DPC backend is not available."
            )
        return queues

    for i in filter_.split(","):
        try:
            raw_queues.append((SyclQueue(i), f"SyclQueue_{i.upper()}"))
        except queue_creation_err:
            pass

    if dtypes is None:
        out = []
        for queue, qid in raw_queues:
            if queue is None and qid is None:
                # Preserve legacy bare-None entry for the CPU host.
                out.append(None)
            else:
                out.append(pytest.param(queue, id=qid))
        return out

    out = []
    for queue, qid in raw_queues:
        base_id = qid if qid is not None else "host"
        for dtype in dtypes:
            if not _queue_supports_dtype(queue, dtype):
                continue
            dt_name = np.dtype(dtype).name
            out.append(pytest.param(queue, dtype, id=f"{base_id}-{dt_name}"))
    return out


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
    if isinstance(targets, str):
        targets = [targets]
    for device in targets:
        try:
            SyclQueue(device)
        except queue_creation_err:
            return False
    return True


def pass_if_not_implemented_for_gpu(reason=""):
    """Decorator for test functions. Asserts the test fails with the specified `reason` when running on GPU.
    Used to ensure that a meaningful error message is provided by the backend."""
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
