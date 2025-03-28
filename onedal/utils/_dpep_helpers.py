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

"""Utilities for Data Parallel Extensions libs, such as DPNP, DPCtl.
See `Data Parallel Extensions for Python <https://github.com/IntelPython/DPEP>`__
"""

import functools

from daal4py.sklearn._utils import _package_check_version


@functools.lru_cache(maxsize=256, typed=False)
def is_dpctl_available(version=None):
    """Checks availability of DPCtl package"""
    try:
        import dpctl
        import dpctl.tensor as dpt

        dpctl_available = True
    except ImportError:
        dpctl_available = False
    if dpctl_available and version is not None:
        dpctl_available = _package_check_version(version, dpctl.__version__)
    return dpctl_available


@functools.lru_cache(maxsize=256, typed=False)
def is_dpnp_available(version=None):
    """Checks availability of DPNP package"""
    try:
        import dpnp

        dpnp_available = True
    except ImportError:
        dpnp_available = False
    if dpnp_available and version is not None:
        dpnp_available = _package_check_version(version, dpnp.__version__)
    return dpnp_available


dpctl_available = is_dpctl_available()
dpnp_available = is_dpnp_available()


if dpnp_available:
    import dpnp
if dpctl_available:
    import dpctl.tensor as dpt


def get_unique_values_with_dpep(X):
    if dpnp_available:
        return dpnp.unique(X)
    elif dpctl_available:
        return dpt.unique_values(X)
    else:
        raise RuntimeError("No DPEP package available to provide `unique` function.")
