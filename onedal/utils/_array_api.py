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

"""Tools to support array_api."""

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from ..utils._third_party import _is_subclass_fast


def _supports_buffer_protocol(obj):
    # the array_api standard mandates conversion with the buffer protocol,
    # which can only be checked via a try-catch in native python
    try:
        memoryview(obj)
    except TypeError:
        return False
    return True


def _asarray(data, xp, *args, **kwargs):
    """Converted input object to array format of xp namespace provided."""
    if hasattr(data, "__array_namespace__") or _supports_buffer_protocol(data):
        return xp.asarray(data, *args, **kwargs)
    elif isinstance(data, Iterable):
        if isinstance(data, tuple):
            result_data = []
            for i in range(len(data)):
                result_data.append(_asarray(data[i], xp, *args, **kwargs))
            data = tuple(result_data)
        else:
            for i in range(len(data)):
                data[i] = _asarray(data[i], xp, *args, **kwargs)
    return data


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {
        "numpy",
        "array_api_compat.numpy",
        "numpy.array_api",
        "sklearn.externals.array_api_compat.numpy",
    }


@lru_cache(100)
def _cls_to_sycl_namespace(cls):
    # use caching to minimize imports, derived from array_api_compat
    if _is_subclass_fast(cls, "dpctl.tensor", "usm_ndarray"):
        import dpctl.tensor as dpt

        return dpt
    elif _is_subclass_fast(cls, "dpnp", "ndarray"):
        import dpnp

        return dpnp
    else:
        raise ValueError(f"SYCL type not recognized: {cls}")


def _get_sycl_namespace(*arrays):
    """Get namespace of sycl arrays."""

    # sycl support designed to work regardless of array_api_dispatch sklearn global value
    sua_iface = {type(x): x for x in arrays if hasattr(x, "__sycl_usm_array_interface__")}

    if len(sua_iface) > 1:
        raise ValueError(f"Multiple SYCL types for array inputs: {sua_iface}")

    if sua_iface:
        (X,) = sua_iface.values()
        return (
            sua_iface,
            _cls_to_sycl_namespace(type(X)),
            hasattr(X, "__array_namespace__"),
        )

    return sua_iface, np, False
