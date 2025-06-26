# ==============================================================================
# Copyright 2024 Intel Corporation
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

import pytest

from onedal import _default_backend as backend
from onedal.tests.utils._device_selection import get_queues
from onedal.utils._third_party import dpctl_available


@pytest.mark.skipif(
    not backend.is_dpc or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
@pytest.mark.parametrize("device_number", [None, 0, 1, 2, 3])
def test_sycl_queue_string_creation(device_type, device_number):
    # create devices from strings
    from dpctl import SyclQueue
    from dpctl._sycl_queue import SyclQueueCreationError

    onedal_SyclQueue = backend.SyclQueue

    device = (
        ":".join([device_type, str(device_number)])
        if device_number is not None
        else device_type
    )

    raised_exception_dpctl = False
    raised_exception_backend = False

    try:
        dpctl_string = SyclQueue(device).sycl_device.filter_string
    except SyclQueueCreationError:
        raised_exception_dpctl = True

    try:
        onedal_string = onedal_SyclQueue(device).sycl_device.filter_string
    except RuntimeError:
        raised_exception_backend = True

    assert raised_exception_dpctl == raised_exception_backend
    if not raised_exception_backend:
        # dpctl filter string converts simple sycl filter_strings
        # i.e. "gpu:1" -> "opencl:gpu:0", use SyclQueue to convert
        # for matching, as oneDAL sycl queue only returns simple
        # strings as these are operationally sufficient
        assert SyclQueue(onedal_string).sycl_device.filter_string == dpctl_string


@pytest.mark.skipif(
    not backend.is_dpc or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("queue", get_queues())
def test_sycl_queue_conversion(queue):
    if queue is None:
        pytest.skip("Not a dpctl queue")
    SyclQueue = queue.__class__
    onedal_SyclQueue = backend.SyclQueue

    q = onedal_SyclQueue(queue)

    # convert back and forth to test `_get_capsule` attribute
    for i in range(10):
        q = SyclQueue(q.sycl_device.filter_string)
        q = onedal_SyclQueue(q)

    assert q.sycl_device.filter_string in queue.sycl_device.filter_string


@pytest.mark.skipif(
    not backend.is_dpc or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("queue", get_queues())
def test_sycl_device_attributes(queue):
    from dpctl import SyclQueue

    if queue is None:
        pytest.skip("Not a dpctl queue")
    onedal_SyclQueue = backend.SyclQueue

    onedal_queue = onedal_SyclQueue(queue)

    # check fp64 support
    assert onedal_queue.sycl_device.has_aspect_fp64 == queue.sycl_device.has_aspect_fp64
    # check fp16 support
    assert onedal_queue.sycl_device.has_aspect_fp16 == queue.sycl_device.has_aspect_fp16
    # check is_cpu
    assert onedal_queue.sycl_device.is_cpu == queue.sycl_device.is_cpu
    # check is_gpu
    assert onedal_queue.sycl_device.is_gpu == queue.sycl_device.is_gpu
    # check device number
    assert onedal_queue.sycl_device.filter_string in queue.sycl_device.filter_string


@pytest.mark.skipif(not backend.is_dpc, reason="requires dpc backend")
def test_backend_queue():
    try:
        q = backend.SyclQueue("cpu")
    except RuntimeError:
        pytest.skip("OpenCL CPU runtime not installed")

    # verify copying via a py capsule object is functional
    q2 = backend.SyclQueue(q._get_capsule())
    # verify copying via the _get_capsule attribute
    q3 = backend.SyclQueue(q)

    q_array = [q, q2, q3]

    assert all([queue.sycl_device.has_aspect_fp64 for queue in q_array])
    assert all([queue.sycl_device.has_aspect_fp16 for queue in q_array])
    assert all([queue.sycl_device.is_cpu for queue in q_array])
    assert all([not queue.sycl_device.is_gpu for queue in q_array])
    assert all(["cpu" in queue.sycl_device.filter_string for queue in q_array])


@pytest.mark.skipif(
    not backend.is_dpc or not dpctl_available, reason="requires dpc backend and dpctl"
)
def test_backend_device_id():
    """verify dpctl device id matches our internal device id"""
    import dpctl

    for i in range(dpctl.get_num_devices()):
        device = dpctl.SyclDevice(str(i))
        if device.is_gpu or device.is_cpu:
            backend_device = backend.SyclDevice(i)
            assert (
                i == backend_device.get_device_id()
            ), f"backend id does not match device_id: {i}"
