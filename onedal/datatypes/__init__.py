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

import numpy as np

from .. import _default_backend
from ._data_conversion import from_table, return_type_constructor, to_table
from ._dlpack import dlpack_to_numpy, get_torch_queue

__all__ = [
    "dlpack_to_numpy",
    "from_table",
    "get_torch_queue",
    "to_table",
    "return_type_constructor",
    "is_onedal_supported_device",
]


def is_onedal_supported_device(arr) -> bool:
    if isinstance(arr, np.ndarray):
        return True
    if not hasattr(arr, "__dlpack_device__"):
        return False
    device = arr.__dlpack_device__()[0]
    return device in [_default_backend.kDLCPU, _default_backend.kDLOneAPI]
