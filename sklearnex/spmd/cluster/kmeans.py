# ==============================================================================
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
# ==============================================================================

from onedal.spmd.cluster import KMeans as onedal_KMeans_SPMD

from ...cluster import KMeans as base_KMeans


class KMeans(base_KMeans):
    def _initialize_onedal_estimator(self):
        """Override to use SPMD backend instead of batch backend."""
        onedal_params = {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": getattr(self, "_n_init", self._resolve_n_init()),
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

        self._onedal_estimator = onedal_KMeans_SPMD(**onedal_params)
