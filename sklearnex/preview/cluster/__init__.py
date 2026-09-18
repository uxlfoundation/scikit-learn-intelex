# ===============================================================================
# Copyright contributors to the oneDAL project
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
# ===============================================================================

from . import hdbscan as _hdbscan

__all__ = []

# '.hdbscan' defines the estimator only when the oneDAL in use provides the
# algorithm, so the version check it does is not repeated here
if hasattr(_hdbscan, "HDBSCAN"):
    from .hdbscan import HDBSCAN

    __all__ += ["HDBSCAN"]
