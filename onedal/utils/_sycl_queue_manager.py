# ==============================================================================
# Copyright 2025 Intel Corporation
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

import threading
from contextlib import contextmanager
from types import SimpleNamespace

from onedal import _default_backend as backend

from .._config import _get_config
from ..datatypes import get_torch_queue
from ._third_party import SyclQueue, is_torch_tensor


class ThreadLocalGlobals:

    def __init__(self):
        self._local = threading.local()
        self._local.queue = None
        self._local.dlpack_queue = {}

        # This special object signifies that the queue system should be
        # disabled. It will force computation to host. This occurs when the
        # thread-local queue is set to this value (and therefore should not be
        # modified).
        self.__fallback_queue = object()
        # Special queue for non-CPU, non-SYCL data associated with dlpack
        self.__non_queue = SimpleNamespace(sycl_device=SimpleNamespace(is_cpu=False))

    # Single instance of thread-local queue.
    # This object as a global within the thread.
    @property
    def queue(self):
        return self._local.queue

    @queue.setter
    def queue(self, value):
        self._local.queue = value

    # dictionary of generic SYCL queues with default SYCL contexts for reuse
    @property
    def dlpack_queue(self) -> "dict[SyclQueue]":
        return self._local.dlpack_queue

    @dlpack_queue.setter
    def dlpack_queue(self, value):
        self._local.dlpack_queue = value


__globals = ThreadLocalGlobals()


def __create_sycl_queue(target):
    if isinstance(target, SyclQueue) or target is None or target is __globals.__non_queue:
        return target
    if isinstance(target, (str, int)):
        return SyclQueue(target)
    raise ValueError(f"Invalid queue or device selector {target=}.")


def get_global_queue():
    """Get the global queue.

    Retrieve it from the config if not set.

    Returns
    -------
    queue: SyclQueue or None
        SYCL Queue object for device code execution. 'None'
        signifies computation on host.
    """
    if (queue := __globals.queue) is not None:
        if queue is __globals.__fallback_queue:
            return None
        return queue

    target = _get_config()["target_offload"]
    if target == "auto":
        # queue will be created from the provided data to each function call
        return None

    q = __create_sycl_queue(target)
    update_global_queue(q)
    return q


def remove_global_queue():
    """Remove the global queue."""
    __globals.queue = None


def update_global_queue(queue):
    """Update the global queue.

    Parameters
    ----------
    queue : SyclQueue or None
        SYCL Queue object for device code execution. None
        signifies computation on host.
    """
    queue = __create_sycl_queue(queue)
    __globals.queue = queue


def fallback_to_host():
    """Enforce a host queue."""
    __globals.queue = __globals.__fallback_queue


def _get_dlpack_queue(obj: object) -> SyclQueue:
    # users should not require direct use of this
    device_type, device_id = obj.__dlpack_device__()
    if device_type == backend.kDLCPU:
        return None
    elif device_type != backend.kDLOneAPI:
        # Data exists on a non-SYCL, non-CPU device. This will trigger an error
        # or a fallback if "fallback_to_host" is set in the config
        return __globals.__non_queue

    if is_torch_tensor(obj):
        return get_torch_queue(obj)
    else:
        # no specialized queue can be extracted. Use or generate a generic. Note,
        # this will behave in unexpected ways for non-default SYCL contexts or
        # with SYCL sub-devices due to limitations in the dlpack standard (not
        # enough info).
        try:
            queue = __globals.__dlpack_queue[device_id]
        except KeyError:
            # use filter string capability to yield a queue
            queue = SyclQueue(str(device_id))
            __globals.__dlpack_queue[device_id] = queue
        return queue


def from_data(*data):
    """Extract the queue from provided data.

    This updates the global queue as well.

    Parameters
    ----------
    *data : arguments
        Data objects which may contain :obj:`dpctl.SyclQueue` objects.

    Returns
    -------
    queue : SyclQueue or None
        SYCL Queue object for device code execution. None
        signifies computation on host.
    """
    for item in data:
        # iterate through all data objects, extract the queue, and verify that all data objects are on the same device
        if usm_iface := getattr(item, "__sycl_usm_array_interface__", None):
            data_queue = usm_iface["syclobj"]
        elif hasattr(item, "__dlpack_device__"):
            data_queue = _get_dlpack_queue(item)
        else:
            data_queue = None

        # no queue, i.e. host data, no more work to do
        if data_queue is None:
            continue

        global_queue = get_global_queue()
        # update the global queue if not set
        if global_queue is None:
            update_global_queue(data_queue)
            global_queue = data_queue

        # if either queue points to a device, assert it's always the same device
        data_dev = data_queue.sycl_device
        global_dev = global_queue.sycl_device
        if (data_dev and global_dev) is not None and data_dev != global_dev:
            # when all data exists on other devices (e.g. not CPU or SYCL devices)
            # failure will come in backend selection occurring in
            # sklearnex._device_offload._get_backend when using __non_queue
            raise ValueError(
                "Data objects are located on different target devices or not on selected device."
            )

    # after we went through the data, global queue is updated and verified (if any queue found)
    return get_global_queue()


@contextmanager
def manage_global_queue(queue, *args):
    """Context manager to manage the global SyclQueue.

    This context manager updates the global queue with the provided queue,
    verifies that all data objects are on the same device, and restores the
    original queue after work is done.

    Parameters
    ----------
    queue : SyclQueue or None
        The queue to set as the global queue. If None,
        the global queue will be determined from the provided data.

    *args : arguments
        Additional data objects to verify their device placement.

    Yields
    ------
    SyclQueue : SyclQueue or None
        The global queue after verification.

    Notes
    -----
        For most applications, the original queue should be ``None``, but
        if there are nested calls to ``manage_global_queue()``, it is
        important to restore the outer queue, rather than setting it to
        ``None``.
    """
    original_queue = get_global_queue()
    try:
        # update the global queue with what is provided, it can be None, then we will get it from provided data
        update_global_queue(queue)
        # find the queues in data to verify that all data objects are on the same device
        yield from_data(*args)
    finally:
        # restore the original queue
        update_global_queue(original_queue)
