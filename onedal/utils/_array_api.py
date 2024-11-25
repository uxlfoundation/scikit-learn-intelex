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
from functools import wraps

from daal4py.sklearn._utils import sklearn_check_version

if sklearn_check_version("1.4"):
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace

import numpy as np
from sklearn import config_context, get_config

from daal4py.sklearn._utils import get_dtype
from daal4py.sklearn._utils import make2d as d4p_make2d
from daal4py.sklearn._utils import sklearn_check_version

from ._dpep_helpers import dpctl_available, dpnp_available

if dpctl_available:
    import dpctl.tensor as dpt
    from dpctl.tensor import usm_ndarray


# TODO:
# move to Array API module.
# TODO
# def make2d(arg, xp=None, is_array_api_compliant=None):
def make2d(arg, xp=None):
    if xp and not _is_numpy_namespace(xp) and arg.ndim == 1:
        return xp.reshape(arg, (arg.size, 1)) if arg.ndim == 1 else arg
    # TODO:
    # reimpl via is_array_api_compliant usage.
    return d4p_make2d(arg)


if dpnp_available:
    import dpnp

    def _convert_to_dpnp(array):
        """Converted input object to dpnp.ndarray format."""
        # Will be removed and `onedal.utils._array_api._asarray` will be
        # used instead after DPNP Array API enabling.
        if isinstance(array, usm_ndarray):
            return dpnp.array(array, copy=False)
        elif isinstance(array, Iterable):
            for i in range(len(array)):
                array[i] = _convert_to_dpnp(array[i])
        return array


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

    if dpctl_available and xp_name in {
        "dpctl.tensor",
    }:
        return dpt.to_numpy(array)
    elif dpnp_available and isinstance(array, dpnp.ndarray):
        return dpnp.asnumpy(array)
    else:
        return _asarray(array, xp)


def _asarray(data, xp, *args, **kwargs):
    """Converted input object to array format of xp namespace provided."""
    if hasattr(data, "__array_namespace__"):
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


def _ravel(array, xp):
    """Return a flattened array.

    Note
    ----
    Input array expected to be contiguous.
    """

    return xp.reshape(array, (array.size,))


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {"numpy", "array_api_compat.numpy", "numpy.array_api"}


def _get_sycl_namespace(*arrays):
    """Get namespace of sycl arrays."""

    # sycl support designed to work regardless of array_api_dispatch sklearn global value
    sua_iface = {type(x): x for x in arrays if hasattr(x, "__sycl_usm_array_interface__")}

    if len(sua_iface) > 1:
        raise ValueError(f"Multiple SYCL types for array inputs: {sua_iface}")

    if sua_iface:
        (X,) = sua_iface.values()

        if hasattr(X, "__array_namespace__"):
            return sua_iface, X.__array_namespace__(), True
        elif dpnp_available and isinstance(X, dpnp.ndarray):
            return sua_iface, dpnp, False
        else:
            raise ValueError(f"SYCL type not recognized: {sua_iface}")

    return sua_iface, None, False


# TODO:
#
sklearn_array_api_version = True


def sklearn_array_api_dispatch(freefunc=False):
    """
    TBD
    """

    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            # if sklearn_array_api_version and not get_config["array_api_dispatch"]:
            if sklearn_array_api_version:
                with config_context(array_api_dispatch=True):
                    return func(obj, *args, **kwargs)
            return func(obj, *args, **kwargs)

        if freefunc:

            @wraps(func)
            def wrapper_free(*args, **kwargs):
                return wrapper_impl(None, *args, **kwargs)

            return wrapper_free

        @wraps(func)
        def wrapper_with_self(self, *args, **kwargs):
            return wrapper_impl(self, *args, **kwargs)

        return wrapper_with_self

    return decorator


if sklearn_check_version("1.5"):

    def get_namespace(*arrays, remove_none=True, remove_types=(str,), xp=None):
        """Get namespace of arrays.

        Extends stock scikit-learn's `get_namespace` primitive to support DPCTL usm_ndarrays
        and DPNP ndarrays.
        If no DPCTL usm_ndarray or DPNP ndarray inputs and backend scikit-learn version supports
        Array API then :obj:`sklearn.utils._array_api.get_namespace` results are drawn.
        Otherwise, numpy namespace will be returned.

        Designed to work for numpy.ndarray, DPCTL usm_ndarrays and DPNP ndarrays without
        `array-api-compat` or backend scikit-learn Array API support.

        For full documentation refer to :obj:`sklearn.utils._array_api.get_namespace`.

        Parameters
        ----------
        *arrays : array objects
            Array objects.

        remove_none : bool, default=True
            Whether to ignore None objects passed in arrays.

        remove_types : tuple or list, default=(str,)
            Types to ignore in the arrays.

        xp : module, default=None
            Precomputed array namespace module. When passed, typically from a caller
            that has already performed inspection of its own inputs, skips array
            namespace inspection.

        Returns
        -------
        usm_iface : TBD

        namespace : module
            Namespace shared by array objects.

        is_array_api : bool
            True of the arrays are containers that implement the Array API spec.
        """

        usm_iface, xp_sycl_namespace, is_array_api_compliant = _get_sycl_namespace(
            *arrays
        )

        if usm_iface:
            return usm_iface, xp_sycl_namespace, is_array_api_compliant
        elif sklearn_check_version("1.4"):
            xp, is_array_api_compliant = sklearn_get_namespace(
                *arrays, remove_none=remove_none, remove_types=remove_types, xp=xp
            )
            return usm_iface, xp, is_array_api_compliant
        else:
            return usm_iface, np, False

else:

    def get_namespace(*arrays):
        """Get namespace of arrays.

        Extends stock scikit-learn's `get_namespace` primitive to support DPCTL usm_ndarrays
        and DPNP ndarrays.
        If no DPCTL usm_ndarray or DPNP ndarray inputs and backend scikit-learn version supports
        Array API then :obj:`sklearn.utils._array_api.get_namespace(*arrays)` results are drawn.
        Otherwise, numpy namespace will be returned.

        Designed to work for numpy.ndarray, DPCTL usm_ndarrays and DPNP ndarrays without
        `array-api-compat` or backend scikit-learn Array API support.

        For full documentation refer to :obj:`sklearn.utils._array_api.get_namespace`.

        Parameters
        ----------
        *arrays : array objects
            Array objects.

        Returns
        -------
        usm_iface : TBD

        namespace : module
            Namespace shared by array objects.

        is_array_api : bool
            True of the arrays are containers that implement the Array API spec.
        """

        usm_iface, xp_sycl_namespace, is_array_api_compliant = _get_sycl_namespace(
            *arrays
        )

        if usm_iface:
            return usm_iface, xp_sycl_namespace, is_array_api_compliant
        elif sklearn_check_version("1.4"):
            xp, is_array_api_compliant = sklearn_get_namespace(*arrays)
            return usm_iface, xp, is_array_api_compliant
        else:
            return usm_iface, np, False
