# ==============================================================================
# Copyright 2021 Intel Corporation
# Copyright 2024 Fujitsu Limited
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

import os

from . import utils
from ._config import config_context, get_config, set_config
from .dispatcher import (
    get_patch_map,
    get_patch_names,
    is_patched_instance,
    patch_sklearn,
    sklearn_is_patched,
    unpatch_sklearn,
)

__all__ = [
    "basic_statistics",
    "cluster",
    "config_context",
    "covariance",
    "decomposition",
    "ensemble",
    "get_config",
    "get_hyperparameters",
    "get_patch_map",
    "get_patch_names",
    "is_patched_instance",
    "linear_model",
    "manifold",
    "metrics",
    "model_selection",
    "neighbors",
    "patch_sklearn",
    "set_config",
    "sklearn_is_patched",
    "svm",
    "unpatch_sklearn",
    "utils",
]
__version__ = "2199.9.9"
onedal_iface_flag = os.environ.get("OFF_ONEDAL_IFACE", "0")
if onedal_iface_flag == "0":
    from onedal import _spmd_backend
    from onedal.common.hyperparameters import get_hyperparameters

    if _spmd_backend is not None:
        __all__.append("spmd")


from ._utils import set_sklearn_ex_verbose

set_sklearn_ex_verbose()
