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

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils._array_api import _asarray, _get_sycl_namespace

if sklearn_check_version("1.2"):
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace
    from sklearn.utils._array_api import _convert_to_numpy as _sklearn_convert_to_numpy

from onedal._device_offload import dpctl_available, dpnp_available

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

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


if sklearn_check_version("1.5"):

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

        usm_iface, xp_sycl_namespace, is_array_api_compliant = _get_sycl_namespace(
            *arrays
        )

        if usm_iface:
            return xp_sycl_namespace, is_array_api_compliant
        elif sklearn_check_version("1.2"):
            return sklearn_get_namespace(
                *arrays, remove_none=remove_none, remove_types=remove_types, xp=xp
            )
        else:
            return np, False

else:

    def get_namespace(*arrays):
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

        usm_iface, xp_sycl_namespace, is_array_api_compliant = _get_sycl_namespace(
            *arrays
        )

        if usm_iface:
            return xp_sycl_namespace, is_array_api_compliant
        elif sklearn_check_version("1.2"):
            return sklearn_get_namespace(*arrays)
        else:
            return np, False
