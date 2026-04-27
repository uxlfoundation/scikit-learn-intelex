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

import numpy as np
from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
from sklearn.metrics import pairwise_distances

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import is_sparse, sklearn_check_version
from onedal.cluster import HDBSCAN as onedal_HDBSCAN

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data


@enable_array_api
@control_n_jobs(decorated_methods=["fit"])
class HDBSCAN(oneDALEstimator, _sklearn_HDBSCAN):
    __doc__ = _sklearn_HDBSCAN.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_HDBSCAN._parameter_constraints}

    _onedal_hdbscan = staticmethod(onedal_HDBSCAN)

    def _onedal_fit(self, X, y=None, queue=None):
        xp, _ = get_namespace(X)
        X = validate_data(
            self, X, accept_sparse=False, dtype=[xp.float64, xp.float32]
        )

        onedal_params = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "alpha": self.alpha,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "n_jobs": self.n_jobs,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "store_centers": self.store_centers,
            "copy": self.copy,
        }
        self._onedal_estimator = self._onedal_hdbscan(**onedal_params)

        self._onedal_estimator.fit(X, queue=queue)
        self.labels_ = self._onedal_estimator.labels_
        self.n_features_in_ = X.shape[1]

        # oneDAL does not compute probabilities; set uniform zeros
        # to match sklearn's interface
        self.probabilities_ = np.zeros(len(self.labels_), dtype=np.float64)

        # Compute cluster centers when requested, since oneDAL does not
        # provide them directly
        if self.store_centers in ("centroid", "both"):
            self.centroids_ = self._compute_centroids(X)
        if self.store_centers in ("medoid", "both"):
            self.medoids_ = self._compute_medoids(X)

    def _compute_centroids(self, X):
        n_clusters = len(set(self.labels_) - {-1})
        centroids = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        for idx in range(n_clusters):
            mask = self.labels_ == idx
            centroids[idx] = np.average(X[mask], weights=self.probabilities_[mask], axis=0)
        return centroids

    def _compute_medoids(self, X):
        metric_params = self.metric_params or {}
        n_clusters = len(set(self.labels_) - {-1})
        medoids = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        for idx in range(n_clusters):
            mask = self.labels_ == idx
            data = X[mask]
            dist_mat = pairwise_distances(data, metric=self.metric, **metric_params)
            dist_mat = dist_mat * self.probabilities_[mask]
            medoid_index = np.argmin(dist_mat.sum(axis=1))
            medoids[idx] = data[medoid_index]
        return medoids

    _onedal_supported_metrics = {
        "euclidean",
        "manhattan",
        "minkowski",
        "chebyshev",
        "cosine",
    }

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.cluster.{class_name}.{method_name}"
        )
        if method_name == "fit":
            X = data[0]
            patching_status.and_conditions(
                [
                    (
                        self.metric in self._onedal_supported_metrics,
                        f"'{self.metric}' metric is not supported. "
                        f"Only {self._onedal_supported_metrics} are supported.",
                    ),
                    (
                        self.cluster_selection_method == "eom",
                        f"'{self.cluster_selection_method}' cluster selection "
                        "method is not supported. Only 'eom' is supported.",
                    ),
                    (
                        self.cluster_selection_epsilon == 0.0,
                        "Non-zero cluster_selection_epsilon is not supported.",
                    ),
                    (
                        self.max_cluster_size is None,
                        "max_cluster_size is not supported.",
                    ),
                    (
                        not self.allow_single_cluster,
                        "allow_single_cluster=True is not supported.",
                    ),
                    (not is_sparse(X), "X is sparse. Sparse input is not supported."),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_HDBSCAN.fit,
            },
            X,
            y,
        )

        return self

    fit.__doc__ = _sklearn_HDBSCAN.fit.__doc__
