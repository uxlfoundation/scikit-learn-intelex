# ==============================================================================
# Copyright 2023 Intel Corporation
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

from . import covariance, ensemble
from .. import onedal_check_version


__all__ = ["covariance", "ensemble"]

if onedal_check_version(2023, 1, 0):
    from . import (
        basic_statistics,
        decomposition,
        linear_model,
        neighbors
    )
    __all__ += [
        "basic_statistics",
        "decomposition",
        "linear_model",
        "neighbors",
    ]
if onedal_check_version(2023, 2, 0):
    from . import cluster
    __all__ += ["cluster"]
