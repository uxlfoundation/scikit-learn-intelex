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

import numpy as np

from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table


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

    def _get_onedal_params(self, dtype=np.float32):
        result_options = "responses"
        if self.store_centers in ("centroid", "both"):
            result_options += "|cluster_centers"
        if self.store_centers in ("medoid", "both"):
            result_options += "|medoid_centers"

        params = {
            "fptype": dtype,
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
            "leaf_size": int(self.leaf_size),
            "store_centers": self.store_centers,
        }

        # the degree is only meaningful for the Minkowski distance
        if self.metric == "minkowski":
            params["degree"] = float(self.degree)

        return params

    @supports_queue
    def fit(self, X, y=None, queue=None):
        X_table = to_table(X, queue=queue)

        params = self._get_onedal_params(X_table.dtype)
        result = self.compute(params, X_table)

        self.labels_ = from_table(result.responses, like=X)[:, 0]
        self.n_clusters_ = int(result.cluster_count)

        # the centers are left empty by oneDAL when it does not find any cluster
        if self.n_clusters_ > 0:
            if self.store_centers in ("centroid", "both"):
                self.centroids_ = from_table(result.cluster_centers, like=X)
            if self.store_centers in ("medoid", "both"):
                self.medoids_ = from_table(result.medoid_centers, like=X)

        return self
