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
import scipy.sparse as sp

from ..utils._third_party import lazy_import


@lazy_import("dpnp")
def _to_dpnp(dpnp, queue, array):
    try:
        return dpnp.asarray(array, usm_type="device", sycl_queue=queue)
    except ValueError as e:
        # ValueError will raise if device does not support the dtype
        # retry with float32 (needed for fp16 and fp64 support issues)
        # try again as float32, if it is a float32 just raise the error.
        if array.dtype == np.float32:
            raise e
        return _to_dpnp(queue, array.astype(np.float32))


def copy_to_dpnp(queue, array):
    if hasattr(array, "tobytes"):
        return _to_dpnp(queue, array)
    else:
        if isinstance(array, Iterable) and not sp.issparse(array):
            array = [copy_to_dpnp(queue, i) for i in array]
        return array
