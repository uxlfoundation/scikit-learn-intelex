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

"""Tools to expose some sklearnex's config settings to onedal4py level."""

import threading

"""
Default values for global configuration parameters.
These values are typically managed through the sklearnex.set_config() interface.
Here we only define the defaults.

target_offload:
    The device primarily used to perform computations.
    If string, expected to be "auto" (the execution context
    is deduced from input data location), or SYCL* filter selector string.
    Global default: "auto".
allow_fallback_to_host:
    If True, allows to fallback computation to host device
    in case particular estimator does not support the selected one.
    Global default: False.
allow_sklearn_after_onedal:
    If True, allows to fallback computation to sklearn after onedal
    backend in case of runtime error on onedal backend computations.
    Global default: True.
use_raw_input:
    If True, uses the raw input data in some SPMD onedal backend computations
    without any checks on data consistency or validity.
    Note: This option is not recommended for general use.
    Global default: False.
"""
_default_global_config = {
    "target_offload": "auto",
    "allow_fallback_to_host": False,
    "allow_sklearn_after_onedal": True,
    "use_raw_input": False,
}

_threadlocal = threading.local()


def _get_onedal_threadlocal_config():
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _default_global_config.copy()
    return _threadlocal.global_config


def _get_config(copy=True):
    """Retrieve current configuration set by :func:`sklearnex.set_config`

    Parameters
    ----------
    copy : bool, default=True
        If 'False', a mutable view of the configuration is returned. Each
        thread has a separate copy of the configuration.

    Returns
    -------
    config : dict
        Keys are parameter names `target_offload` and
        `allow_fallback_to_host` that can be passed
        to :func:`sklearnex.set_config`.
    """
    onedal_config = _get_onedal_threadlocal_config()
    if copy:
        onedal_config = onedal_config.copy()
    return onedal_config
