# ==============================================================================
# Copyright Contributors to the oneDAL Project
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

from collections.abc import Iterable

import numpy as np

from onedal import _default_backend as backend

from ..utils._third_party import convert_sklearnex_queue, lazy_import

cpu_dlpack_device = (backend.kDLCPU, 0)


@lazy_import("torch.xpu")
@convert_sklearnex_queue
def get_torch_queue(torchxpu, array):
    return backend.SyclQueue(torchxpu.current_stream(array.get_device()).sycl_queue)


def dlpack_to_numpy(obj, device):
    # check dlpack data location.
    if device() != cpu_dlpack_device:
        if hasattr(item, "to_device"):
            # use of the "cpu" string as device not officially part of
            # the array api standard but widely supported
            item = obj.to_device("cpu")
        elif hasattr(obj, "to"):
            # pytorch-specific fix as it is not array api compliant
            obj = obj.to("cpu")
        else:
            raise TypeError(f"cannot move {type(obj)} to cpu")

    # convert to numpy
    if hasattr(obj, "__array__"):
        # `copy`` param for the `asarray`` is not set.
        # The object is copied only if needed
        obj = np.asarray(obj)
    else:
        # requires numpy 1.23
        obj = np.from_dlpack(obj)
    return obj
