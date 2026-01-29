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

import sys
import warnings
from contextlib import contextmanager

from sklearn import get_config as skl_get_config
from sklearn import set_config as skl_set_config

from daal4py.sklearn._utils import sklearn_check_version
from onedal._config import _get_config as onedal_get_config

__all__ = ["get_config", "set_config", "config_context"]

tab = "    " if (sys.version_info.major == 3 and sys.version_info.minor < 13) else ""
_options_docstring = f"""Parameters
{tab}----------
{tab}target_offload : str or dpctl.SyclQueue or None
{tab}    The device used to perform computations, either as a string indicating a name
{tab}    recognized by the SyCL runtime, such as ``"gpu"``, ``"gpu:0"``, or as a
{tab}    :obj:`dpctl.SyclQueue` object indicating where to move the data.
{tab}
{tab}    Assuming SyCL-related dependencies are installed, the list of devices recognized
{tab}    by SyCL can be retrieved through the CLI tool ``sycl-ls`` in a shell, or through
{tab}    :obj:`dpctl.get_devices` in a Python process.
{tab}
{tab}    String ``"auto"`` is also accepted.
{tab}
{tab}    Global default: ``"auto"``.
{tab}
{tab}allow_fallback_to_host : bool or None
{tab}    If ``True``, allows computations to fall back to host device (CPU) when an unsupported
{tab}    operation is attempted on GPU through ``target_offload``.
{tab}
{tab}    Global default: ``False``.
{tab}
{tab}allow_sklearn_after_onedal : bool or None, default=None
{tab}    If ``True``, allows computations to fall back to stock scikit-learn when no
{tab}    accelered version of the operation is available (see :ref:`algorithms`).
{tab}
{tab}    Global default: ``True``.
{tab}
{tab}use_raw_input : bool or None
{tab}    If ``True``, uses the raw input data in some SPMD onedal backend computations
{tab}    without any checks on data consistency or validity. Note that this can be
{tab}    better achieved through usage of :ref:`array API classes <array_api>` without
{tab}    ``target_offload``. Not recommended for general use.
{tab}
{tab}    Global default: ``False``.
{tab}
{tab}    .. deprecated:: 2026.0
{tab}
{tab}sklearn_configs : kwargs
{tab}    Other settings accepted by scikit-learn. See :obj:`sklearn.set_config` for
{tab}    details.
{tab}
{tab}Warnings
{tab}--------
{tab}Using ``use_raw_input=True`` is not recommended for general use as it
{tab}bypasses data consistency checks, which may lead to unexpected behavior. It is
{tab}recommended to use the newer :ref:`array API <array_api>` instead.
{tab}
{tab}Note
{tab}----
{tab}Usage of ``target_offload`` requires additional dependencies - see
{tab}:ref:`GPU support <oneapi_gpu>` for more information."""


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
):  # numpydoc ignore=PR01,PR07
    """Set global configuration.

    %_options_docstring%

    See Also
    --------
    config_context : Context manager for global configuration.
    get_config : Retrieve current values of the global configuration.
    """

    skl_set_config(**sklearn_configs)

    local_config = onedal_get_config(copy=False)

    if target_offload is not None:
        local_config["target_offload"] = target_offload
    if allow_fallback_to_host is not None:
        local_config["allow_fallback_to_host"] = allow_fallback_to_host
    if allow_sklearn_after_onedal is not None:
        local_config["allow_sklearn_after_onedal"] = allow_sklearn_after_onedal
    if use_raw_input is not None:
        warnings.warn(
            "The 'use_raw_input' parameter is deprecated and will be removed in version 2026.0. "
            "On-device input validation can now be achieved by setting 'array_api_dispatch' to True.",
            FutureWarning,
            stacklevel=2,
        )
        local_config["use_raw_input"] = use_raw_input


set_config.__doc__ = set_config.__doc__.replace(
    "%_options_docstring%", _options_docstring
)


@contextmanager
def config_context(**new_config):  # numpydoc ignore=PR01,PR07
    """Context manager for local scikit-learn-intelex configurations.

    %_options_docstring%

    Note
    ----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.
    """
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)


config_context.__doc__ = config_context.__doc__.replace(
    "%_options_docstring%", _options_docstring
)
