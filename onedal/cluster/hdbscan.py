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

from onedal._device_offload import supports_queue

from ..common._backend import bind_default_backend
from ..common._mixin import ClusterMixin
from ..datatypes import from_table, to_table


class HDBSCAN(ClusterMixin):
    def __init__(
        self,
        min_cluster_size=5,
        *,
        min_samples=None,
        metric="euclidean",
        metric_params=None,
        alpha=1.0,
        algorithm="auto",
        leaf_size=40,
        n_jobs=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        store_centers=None,
        copy=False,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy

    @bind_default_backend("hdbscan.clustering")
    def compute(self, params, data_table): ...

    def _get_onedal_params(self, dtype=np.float32):
        min_samples = (
            self.min_samples if self.min_samples is not None else self.min_cluster_size
        )

        # Map sklearn algorithm to oneDAL method
        method = "brute_force"
        if self.algorithm in ("kd_tree", "kdtree"):
            method = "kd_tree"
        elif self.algorithm == "auto":
            # auto: use kd_tree for low-dim euclidean-family, brute_force otherwise
            if self.metric in ("euclidean", "manhattan", "minkowski", "chebyshev"):
                method = "kd_tree"
            else:
                method = "brute_force"

        params = {
            "fptype": dtype,
            "method": method,
            "min_cluster_size": int(self.min_cluster_size),
            "min_samples": int(min_samples),
            "metric": self.metric,
            "result_options": "responses",
            "cluster_selection": self.cluster_selection_method,
            "allow_single_cluster": bool(self.allow_single_cluster),
        }

        # Set degree for Minkowski metric
        if self.metric == "minkowski":
            p = 2.0  # default
            if self.metric_params is not None and "p" in self.metric_params:
                p = float(self.metric_params["p"])
            params["degree"] = p

        return params

    @supports_queue
    def fit(self, X, y=None, queue=None):
        X_table = to_table(X, queue=queue)

        params = self._get_onedal_params(X_table.dtype)
        result = self.compute(params, X_table)

        # 2d table but only 1d of information
        self.labels_ = from_table(result.responses, like=X)[:, 0].astype(np.intp)
        self.n_clusters_ = int(result.cluster_count)
        return self
