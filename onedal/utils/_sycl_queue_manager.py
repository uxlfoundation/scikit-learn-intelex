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

from contextlib import contextmanager

from .._config import _get_config
from ._third_party import SyclQueue

# This special object signifies that the queue system should be
# disabled. It will force computation to host. This occurs when the
# global queue is set to this value (and therefore should not be
# modified).
__fallback_queue = object()
# single instance of global queue
__global_queue = None


def __create_sycl_queue(target):
    if isinstance(target, SyclQueue) or target is None:
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
    if (queue := __global_queue) is not None:
        if SyclQueue:
            if queue is __fallback_queue:
                return None
            elif not isinstance(queue, SyclQueue):
                raise ValueError("Global queue is not a SyclQueue object.")
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
    global __global_queue
    __global_queue = None


def update_global_queue(queue):
    """Update the global queue.

    Parameters
    ----------
    queue : SyclQueue or None
        SYCL Queue object for device code execution. None
        signifies computation on host.
    """
    global __global_queue
    queue = __create_sycl_queue(queue)
    __global_queue = queue


def fallback_to_host():
    """Enforce a host queue."""
    global __global_queue
    __global_queue = __fallback_queue


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
        try:
            usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        except RuntimeError as e:
            if "SUA interface" in str(e):
                # ignore SUA interface errors and move on
                continue
            else:
                # unexpected, re-raise
                raise e

        if usm_iface is None:
            # no interface found - try next data object
            continue

        # extract the queue
        global_queue = get_global_queue()
        data_queue = usm_iface["syclobj"]
        if not data_queue:
            # no queue, i.e. host data, no more work to do
            continue

        # update the global queue if not set
        if global_queue is None:
            update_global_queue(data_queue)
            global_queue = data_queue

        # if either queue points to a device, assert it's always the same device
        data_dev = data_queue.sycl_device
        global_dev = global_queue.sycl_device
        if (data_dev and global_dev) is not None and data_dev != global_dev:
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
