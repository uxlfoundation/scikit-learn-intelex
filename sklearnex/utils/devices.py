# ==============================================================================
# Copyright contributors to the oneDAL project
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

from onedal import _default_backend


def check_data_is_allocated_on_device(arr) -> bool:
    """Check if some array input is allocated on a SYCL device.

    Parameters
    ----------
    arr : array
        Array object from an array library like NumPy or DPNP.

    Returns
    -------
    bool
        Whether the data is allocated on a SYCL device or not.

    Notes
    -----
    In general, this should be handled through the global queue.
    This is only for cases where no global queue is used (e.g.
    `move_estimator_to`)."""
    if not hasattr(arr, "__dlpack_device__"):
        return False
    device = arr.__dlpack_device__()
    if device is None:
        return False
    return device[0] == _default_backend.kDLOneAPI
