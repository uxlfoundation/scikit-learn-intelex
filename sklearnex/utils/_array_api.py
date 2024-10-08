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

import itertools

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils._array_api import _asarray, _get_sycl_namespace

# TODO:
# check the version of skl.
if sklearn_check_version("1.2"):
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace
    from sklearn.utils._array_api import _convert_to_numpy as _sklearn_convert_to_numpy

from onedal._device_offload import dpctl_available, dpnp_available

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp

_NUMPY_NAMESPACE_NAMES = {"numpy", "array_api_compat.numpy"}


def yield_namespaces(include_numpy_namespaces=True):
    """Yield supported namespace.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    include_numpy_namespaces : bool, default=True
        If True, also yield numpy namespaces.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.
    """
    for array_namespace in [
        # The following is used to test the array_api_compat wrapper when
        # array_api_dispatch is enabled: in particular, the arrays used in the
        # tests are regular numpy arrays without any "device" attribute.
        "numpy",
        # Stricter NumPy-based Array API implementation. The
        # array_api_strict.Array instances always have a dummy "device" attribute.
        "array_api_strict",
        "dpctl.tensor",
        "cupy",
        "torch",
    ]:
        if not include_numpy_namespaces and array_namespace in _NUMPY_NAMESPACE_NAMES:
            continue
        yield array_namespace


def yield_namespace_device_dtype_combinations(include_numpy_namespaces=True):
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Parameters
    ----------
    include_numpy_namespaces : bool, default=True
        If True, also yield numpy namespaces.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in yield_namespaces(
        include_numpy_namespaces=include_numpy_namespaces
    ):
        if array_namespace == "torch":
            for device, dtype in itertools.product(
                ("cpu", "cuda"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
            yield array_namespace, "mps", "float32"
        elif array_namespace == "dpctl.tensor":
            for device, dtype in itertools.product(
                ("cpu", "gpu"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
        else:
            yield array_namespace, None, None


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

    # if dpctl_available and isinstance(array, dpctl.tensor):
    if dpctl_available and xp_name in {
        "dpctl.tensor",
    }:
        return dpt.to_numpy(array)
    elif dpnp_available and isinstance(array, dpnp.ndarray):
        return dpnp.asnumpy(array)
    elif sklearn_check_version("1.2"):
        return _sklearn_convert_to_numpy(array, xp)
    else:
        return _asarray(array, xp)


def get_namespace(*arrays, remove_none=True, remove_types=(str,), xp=None):
    """Get namespace of arrays.

    TBD

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    Returns
    -------
    namespace : module
        Namespace shared by array objects.

    is_array_api : bool
        True of the arrays are containers that implement the Array API spec.
    """

    sycl_type, xp_sycl_namespace, is_array_api_compliant = _get_sycl_namespace(*arrays)

    if sycl_type:
        return xp_sycl_namespace, is_array_api_compliant
    elif sklearn_check_version("1.2"):
        return sklearn_get_namespace(
            *arrays, remove_none=remove_none, remove_types=remove_types, xp=xp
        )
    else:
        return np, False
