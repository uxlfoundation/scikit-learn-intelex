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
    """Retrieve current values for configuration set
    by :func:`sklearnex.set_config`
    Parameters
    ----------
    copy : bool, default=True
        If False, the values ​​of the global config are returned,
        which can further be overwritten.
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
