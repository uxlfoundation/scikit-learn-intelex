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

"""Utilities for accessing third party pacakges such as DPNP, DPCtl.

Notes
-----
See `Data Parallel Extensions for Python <https://github.com/IntelPython/DPEP>`__
"""

import functools
import importlib
import sys
from typing import Callable

from daal4py.sklearn._utils import _package_check_version


@functools.lru_cache(maxsize=256, typed=False)
def is_dpctl_available(version=None):
    """Check availability of DPCtl package.

    Parameters
    ----------
    version : str or None, default=None
        Minimum supported dpctl version if installed.
        Secondary version check skipped if set to None.

    Returns
    -------
    dpctl_available : bool
        Flag describing import success.
    """
    try:
        import dpctl

        dpctl_available = True
    except ImportError:
        dpctl_available = False
    if dpctl_available and version is not None:
        dpctl_available = _package_check_version(version, dpctl.__version__)
    return dpctl_available


# Note: The dpctl package contains both SYCL infrastructure as well as a
# data framework (dpctl.tensor). dpctl.tensor is not imported when dpctl is
# imported. All data frameworks are to be lazy-loaded, but aspects of dpctl
# (e.g. SyclQueue) are loaded as normal as it is preferred over included
# backend replacements in the core onedal python module.
dpctl_available = is_dpctl_available()

if dpctl_available:
    from dpctl import SyclQueue
else:
    from onedal import _dpc_backend

    SyclQueue = getattr(_dpc_backend, "SyclQueue", None)


def lazy_import(module_name: str) -> Callable:
    """Lazy load a python module for use in a function.

    Decorator which uses dependency injection with monkeypatching to import
    a python module when called. This is done only once on first usage of
    the wrapped function to minimize overhead and reduce branching.

    Parameters
    ----------
    module_name : str
        Name of the module to be imported via importlib.

    Returns
    -------
    decorator : callable
        Flag describing import success.

    Notes
    -----
    The wrapped original function should have the module as the first
    argument. This will be hidden to the user. Lazy imports can be stacked
    for multiple
    """

    # func should have leading arguments which are following a dependency
    # injection paradigm
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*first_args, **first_kwargs):
            mod = importlib.import_module(module_name)

            # hide dependency injection of the original function
            def real_func(*args, **kwargs):
                return func(mod, *args, **kwargs)

            # Monkeypatch the original in a general fashion (cannot use
            # globals())
            modname = func.__module__
            funcname = func.__name__
            setattr(sys.modules[modname], funcname, real_func)

            return real_func(*first_args, **first_kwargs)

        return wrapper

    return decorator


@functools.lru_cache(100)
def _is_subclass_fast(cls: type, modname: str, clsname: str) -> bool:
    # Taken directly from array_api_compat.common._helpers to use for
    # general use (not just array_api)
    try:
        mod = sys.modules[modname]
    except KeyError:
        return False
    parent_cls = getattr(mod, clsname)
    return issubclass(cls, parent_cls)


def is_dpnp_array(x):
    """Return True if 'x' is a dpnp array

    This function does not import dpnp if it has not already been imported
    and is therefore cheap to use.
    """
    return _is_subclass_fast(type(x), "dpnp", "array")


def is_dpctl_tensor(x):
    """Return True if 'x' is a dpnp array

    This function does not import dpnp if it has not already been imported
    and is therefore cheap to use.
    """
    return _is_subclass_fast(type(x), "dpctl.tensor", "usm_ndarray")


def get_unique_values_with_dpep(X):
    if dpnp_available:
        return dpnp.unique(X)
    elif dpctl_available:
        return dpt.unique_values(X)
    else:
        raise RuntimeError("No DPEP package available to provide `unique` function.")
