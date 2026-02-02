# ==============================================================================
# Copyright 2025 Intel Corporation
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

from onedal.spmd.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier
from onedal.spmd.neighbors import KNeighborsRegressor as onedal_KNeighborsRegressor
from onedal.spmd.neighbors import NearestNeighbors as onedal_NearestNeighbors

from ...neighbors import KNeighborsClassifier as base_KNeighborsClassifier
from ...neighbors import KNeighborsRegressor as base_KNeighborsRegressor
from ...neighbors import NearestNeighbors as base_NearestNeighbors


class KNeighborsClassifier(base_KNeighborsClassifier):
    _onedal_estimator = onedal_KNeighborsClassifier

    def _onedal_predict(self, X, queue=None):
        """Override to call SPMD estimator predict directly."""
        return self._onedal_estimator.predict(X, queue=queue)


class KNeighborsRegressor(base_KNeighborsRegressor):
    _onedal_estimator = onedal_KNeighborsRegressor

    def _onedal_predict(self, X, queue=None):
        """Override to call SPMD estimator predict directly."""
        return self._onedal_estimator.predict(X, queue=queue)


class NearestNeighbors(base_NearestNeighbors):
    _onedal_estimator = onedal_NearestNeighbors
