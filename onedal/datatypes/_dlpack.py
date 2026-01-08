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

import numpy as np

from onedal import _default_backend as backend

from ..utils._third_party import convert_sklearnex_queue, lazy_import

cpu_dlpack_device = (backend.kDLCPU, 0)


@lazy_import("torch.xpu")
@convert_sklearnex_queue
def get_torch_queue(torchxpu, array):
    return backend.SyclQueue(torchxpu.current_stream(array.get_device()).sycl_queue)


def dlpack_to_numpy(obj):
    # check dlpack data location.
    if obj.__dlpack_device__() != cpu_dlpack_device:
        if hasattr(obj, "to_device"):
            # use of the "cpu" string as device not officially part of
            # the array api standard but widely supported
            obj = obj.to_device("cpu")
        elif hasattr(obj, "to"):
            # pytorch-specific fix as it is not array api compliant
            obj = obj.to("cpu")
        else:
            raise TypeError(f"cannot move {type(obj)} to cpu")

    # convert to numpy
    try:
        # Some frameworks implement an __array__ method just to
        # throw a RuntimeError when used (array_api_strict, dpctl),
        # or a TypeError (array_api-strict) rather than an AttributeError
        # therefore a try catch is necessary (logic is essentially a
        # getattr call + some)
        obj = obj.__array__()
    except (AttributeError, RuntimeError, TypeError):
        # requires numpy 1.23
        try:
            obj = np.from_dlpack(obj)
        except AttributeError:
            raise NotImplementedError(
                "Upgrade NumPy >= 1.23 for dlpack support"
            ) from None

    return obj
