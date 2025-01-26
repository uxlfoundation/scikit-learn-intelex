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

try:
    import array_api_compat

    array_api_available = True
except ImportError:
    array_api_available = False

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils._array_api import _get_sycl_namespace

from .._config import get_config

if sklearn_check_version("1.2"):
    from sklearn.utils._array_api import get_namespace as sklearn_get_namespace


def get_namespace(*arrays):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API
    compatible namespace object, if any. NumPy 1.22 and later can
    construct such containers using the `numpy.array_api` namespace
    for instance.

    This function will return the namespace of SYCL-related arrays
    which define the __sycl_usm_array_interface__ attribute
    regardless of array_api support, the configuration of
    array_api_dispatch, or scikit-learn version.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the
    `_NumPyApiWrapper` compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it
    call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyApiWrapper`
    compatibility wrapper is always returned irrespective of
    the fact that arrays implement the `__array_namespace__`
    protocol or not.

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

    sycl_type, xp, is_array_api_compliant = _get_sycl_namespace(*arrays)

    if sycl_type:
        return xp, is_array_api_compliant
    # TODO:
    # correct condition.
    # scikit-learn's get_namespace require config_contex(array_api_dispatch=True).
    # elif sklearn_check_version("1.2") and get_config()["array_api_dispatch"]:
    elif sklearn_check_version("1.2"):
        namespace, is_array_api_compliant = sklearn_get_namespace(*arrays)
        return namespace, is_array_api_compliant
    # TODO:
    # on PR 2096.
    # elif array_api_available:
    #     namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True
    #     return namespace, is_array_api_compliant
    # TODO:
    # should be removed.
    else:
        return np, False
