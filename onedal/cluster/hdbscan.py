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

from .. import onedal_check_version
from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table

# HDBSCAN was added to oneDAL in 2026.2, and 'onedal/cluster/hdbscan.cpp' is compiled
# out below that version, so the 'hdbscan' backend submodule does not exist. The class
# is gated rather than only its import in 'onedal/cluster/__init__.py' because
# 'bind_default_backend' resolves the backend method while the class body is executed:
# without the gate the module would raise on a direct import, which is what
# 'onedal/tests/test_common.py::test_relative_importing' does for every onedal module.
if onedal_check_version(2026, 2, 0):

    class HDBSCAN:
        # all parameters follow oneDAL's naming and semantics, the translation of
        # scikit-learn's parameters happens in the sklearnex estimator
        def __init__(
            self,
            min_cluster_size=5,
            *,
            min_samples=5,
            metric="euclidean",
            degree=2.0,
            alpha=1.0,
            method="by_default",
            leaf_size=40,
            cluster_selection="eom",
            allow_single_cluster=False,
            cluster_selection_epsilon=0.0,
            max_cluster_size=0,
            store_centers="none",
        ):
            self.min_cluster_size = min_cluster_size
            self.min_samples = min_samples
            self.metric = metric
            self.degree = degree
            self.alpha = alpha
            self.method = method
            self.leaf_size = leaf_size
            self.cluster_selection = cluster_selection
            self.allow_single_cluster = allow_single_cluster
            self.cluster_selection_epsilon = cluster_selection_epsilon
            self.max_cluster_size = max_cluster_size
            self.store_centers = store_centers

        @bind_default_backend("hdbscan.clustering")
        def compute(self, params, data_table): ...

        def _get_onedal_params(self, data):
            result_options = "responses"
            if self.store_centers in ("centroid", "both"):
                result_options += "|cluster_centers"
            if self.store_centers in ("medoid", "both"):
                result_options += "|medoid_centers"

            return {
                "fptype": data.dtype,
                "method": self.method,
                "min_cluster_size": int(self.min_cluster_size),
                "min_samples": int(self.min_samples),
                "metric": self.metric,
                "result_options": result_options,
                "cluster_selection": self.cluster_selection,
                "allow_single_cluster": bool(self.allow_single_cluster),
                "cluster_selection_epsilon": float(self.cluster_selection_epsilon),
                "max_cluster_size": int(self.max_cluster_size),
                "alpha": float(self.alpha),
                "degree": float(self.degree),
                "leaf_size": int(self.leaf_size),
                "store_centers": self.store_centers,
            }

        @supports_queue
        def fit(self, X, y=None, queue=None):
            X_table = to_table(X, queue=queue)

            params = self._get_onedal_params(X_table)
            result = self.compute(params, X_table)

            # 2d table but only 1d of information
            self.labels_ = from_table(result.responses, like=X)[:, 0]
            self.n_clusters_ = int(result.cluster_count)

            # oneDAL computes the centers only when it is asked to, and leaves them
            # empty when it does not find any cluster, 'None' marks both absences
            self.centroids_ = None
            self.medoids_ = None
            if self.n_clusters_ > 0:
                if self.store_centers in ("centroid", "both"):
                    self.centroids_ = from_table(result.cluster_centers, like=X)
                if self.store_centers in ("medoid", "both"):
                    self.medoids_ = from_table(result.medoid_centers, like=X)

            return self
