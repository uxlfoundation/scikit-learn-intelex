# ==============================================================================
# Copyright 2021 Intel Corporation
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

from contextlib import contextmanager
from os import environ

from sklearn import get_config as skl_get_config
from sklearn import set_config as skl_set_config

from daal4py.sklearn._utils import sklearn_check_version
from onedal._config import _get_config as onedal_get_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global configuration.
    set_config : Set global configuration.
    """
    sklearn = skl_get_config()
    sklearnex = onedal_get_config()
    return {**sklearn, **sklearnex}


def set_config(
    target_offload=None,
    allow_fallback_to_host=None,
    allow_sklearn_after_onedal=None,
    use_raw_input=None,
    **sklearn_configs,
):
    """Set global configuration.

    Parameters
    ----------
    target_offload : str or SyclQueue or None, default=None
        The device primarily used to perform computations.
        If string, expected to be "auto" (the execution context
        is deduced from input data location),
        or SYCL* filter selector string. Global default: "auto".

    allow_fallback_to_host : bool or None, default=None
        If True, allows to fallback computation to host device
        in case particular estimator does not support the selected one.
        Global default: False.

    allow_sklearn_after_onedal : bool or None, default=None
        If True, allows to fallback computation to sklearn after onedal
        backend in case of runtime error on onedal backend computations.
        Global default: True.

    use_raw_input : bool or None, default=None
        If True, uses the raw input data in some SPMD onedal backend computations
        without any checks on data consistency or validity.
        Not recommended for general use.
        Global default: False.

        .. deprecated:: 2026.0

    **sklearn_configs : kwargs
        Scikit-learn configuration settings dependent on the installed version
        of scikit-learn.

    See Also
    --------
    config_context : Context manager for global configuration.
    get_config : Retrieve current values of the global configuration.

    Warnings
    --------
    Using ``use_raw_input=True`` is not recommended for general use as it
    bypasses data consistency checks, which may lead to unexpected behavior.

    Use of ``target_offload`` requires the DPC++ backend. Setting a
    non-default value (e.g ``cpu`` or ``gpu``) without this backend active
    will raise an error.
    """

    array_api_dispatch = sklearn_configs.get("array_api_dispatch", False)
    if array_api_dispatch and sklearn_check_version("1.6"):
        environ["SCIPY_ARRAY_API"] = "1"

    skl_set_config(**sklearn_configs)

    local_config = onedal_get_config(copy=False)

    if target_offload is not None:
        local_config["target_offload"] = target_offload
    if allow_fallback_to_host is not None:
        local_config["allow_fallback_to_host"] = allow_fallback_to_host
    if allow_sklearn_after_onedal is not None:
        local_config["allow_sklearn_after_onedal"] = allow_sklearn_after_onedal
    if use_raw_input is not None:
        local_config["use_raw_input"] = use_raw_input


@contextmanager
def config_context(**new_config):  # numpydoc ignore=PR01,PR07
    """Context manager for global scikit-learn configuration.

    Parameters
    ----------
    target_offload : str or SyclQueue or None, default=None
        The device primarily used to perform computations.
        If string, expected to be "auto" (the execution context
        is deduced from input data location),
        or SYCL* filter selector string. Global default: "auto".

    allow_fallback_to_host : bool or None, default=None
        If True, allows to fallback computation to host device
        in case particular estimator does not support the selected one.
        Global default: False.

    allow_sklearn_after_onedal : bool or None, default=None
        If True, allows to fallback computation to sklearn after onedal
        backend in case of runtime error on onedal backend computations.
        Global default: True.

    use_raw_input : bool or None, default=None
        .. deprecated:: 2026.0
        If True, uses the raw input data in some SPMD onedal backend computations
        without any checks on data consistency or validity.
        Not recommended for general use.
        Global default: False.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.

    Warnings
    --------
    Using ``use_raw_input=True`` is not recommended for general use as it
    bypasses data consistency checks, which may lead to unexpected behavior.
    """
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
