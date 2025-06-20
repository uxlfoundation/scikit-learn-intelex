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

"""Utilities for accessing third party packages such as DPNP, DPCtl.

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
    from onedal import _default_backend as backend

    SyclQueue = backend.SyclQueue


def lazy_import(*module_names: str) -> Callable:
    """Lazy load python modules for use in a function.

    Decorator which uses dependency injection with monkeypatching to import
    python modules when called. This is done only once on first usage of
    the wrapped function to minimize overhead and reduce branching.

    Parameters
    ----------
    *module_names : strings
        Arguments are names of modules to be imported via importlib.

    Returns
    -------
    decorator : callable
        Flag describing import success.

    Notes
    -----
    The wrapped original function should have the modules as the first
    arguments. This will be hidden to the user. Lazy imports cannot be
    stacked, instead pass multiple arguments to the decorator.
    """

    # func should have leading arguments which are following a dependency
    # injection paradigm
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*first_args, **first_kwargs):
            modules = tuple(importlib.import_module(i) for i in module_names)

            # hide dependency injection of the original function
            # onedal/sklearnex will call the underlying function
            # via ``public_func``
            def public_func(*args, **kwargs):
                return func(*modules, *args, **kwargs)

            # Monkeypatch the original in a general fashion (cannot use
            # globals())
            modname = func.__module__
            funcname = func.__name__
            setattr(sys.modules[modname], funcname, public_func)

            return public_func(*first_args, **first_kwargs)

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


def is_dpnp_ndarray(x: object) -> bool:
    """Return True if 'x' is a dpnp ndarray.

    This function does not import dpnp if it has not already been imported
    and is therefore cheap to use.

    Parameters
    ----------
    x : object
        Any python object.

    Returns
    -------
    is_dpnp : bool
        Flag if subclass of dpnp.ndarray.
    """
    return _is_subclass_fast(type(x), "dpnp", "ndarray")


def is_dpctl_tensor(x: object) -> bool:
    """Return True if 'x' is a dpctl usm_ndarray.

    This function does not import dpnp if it has not already been imported
    and is therefore cheap to use.

    Parameters
    ----------
    x : object
        Any python object.

    Returns
    -------
    is_dpctl : bool
        Flag if subclass of dpctl.tensor.usm_ndarray.
    """
    return _is_subclass_fast(type(x), "dpctl.tensor", "usm_ndarray")
