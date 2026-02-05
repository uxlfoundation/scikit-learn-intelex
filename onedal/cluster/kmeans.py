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

import warnings
from abc import ABC

import numpy as np

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.basic_statistics import BasicStatistics
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

if daal_check_version((2023, "P", 200)):
    from .kmeans_init import KMeansInit

from sklearn.cluster._kmeans import _kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from .._config import _get_config
from ..common._mixin import ClusterMixin, TransformerMixin
from ..datatypes import from_table, to_table
from ..utils.validation import _check_array, _is_arraylike_not_scalar, _is_csr


class _BaseKMeans(TransformerMixin, ClusterMixin, ABC):
    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
        n_local_trials=None,
        algorithm="lloyd",
    ):
        # __init__ only stores user-visible params
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.n_local_trials = n_local_trials
        self.algorithm = algorithm  # kept for parity; we support "lloyd" only

        # runtime/learned attrs (set during fit)
        self._tol = None
        self.model_ = None
        self.n_iter_ = None
        self.inertia_ = None
        self.labels_ = None
        self.n_features_in_ = None
        self._cluster_centers_ = None

    # --- pybind11 backends (thin proxies) ---

    @bind_default_backend("kmeans_common", no_policy=True)
    def _is_same_clustering(self, labels, best_labels, n_clusters): ...

    @bind_default_backend("kmeans.clustering")
    def train(self, params, X_table, centroids_table): ...

    @bind_default_backend("kmeans.clustering")
    def infer(self, params, model, X_table): ...

    # --- helpers matching the pattern ---

    def _get_basic_statistics_backend(self, result_options):
        return BasicStatistics(result_options)

    def _tolerance(self, X_table, rtol, is_csr, dtype):
        if rtol == 0.0:
            return 0.0
        dummy = to_table(None)
        bs = self._get_basic_statistics_backend("variance")
        res = bs._compute_raw(X_table, dummy, dtype, is_csr)
        mean_var = from_table(res.variance).mean()
        return mean_var * rtol

    def _compute_tolerance(self, X_table, is_csr, dtype):
        """Compute absolute tolerance from relative tolerance using data variance."""
        self._tol = self._tolerance(X_table, self.tol, is_csr, dtype)

    def _get_onedal_params(self, is_csr=False, dtype=np.float32, result_options=None):
        thr = self._tol if self._tol is not None else self.tol
        return {
            # fptype chosen from input table dtype (pattern)
            "fptype": dtype,
            # map method names to backend dispatch (CSR vs dense)
            "method": "lloyd_csr" if is_csr else "by_default",
            "seed": -1,
            "max_iteration_count": self.max_iter,
            "cluster_count": self.n_clusters,
            "accuracy_threshold": thr,
            "result_options": "" if result_options is None else result_options,
        }

    def _get_kmeans_init(self, cluster_count, seed, algorithm, is_csr):
        return KMeansInit(
            cluster_count=cluster_count,
            seed=seed,
            algorithm=algorithm,
            is_csr=is_csr,
        )

    def _init_centroids_onedal(
        self,
        X_table,
        init,
        random_seed,
        is_csr,
        dtype=np.float32,
        n_centroids=None,
    ):
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if isinstance(init, str) and init == "k-means++":
            algorithm = "plus_plus_dense" if not is_csr else "plus_plus_csr"
        elif isinstance(init, str) and init == "random":
            algorithm = "random_dense" if not is_csr else "random_csr"
        elif _is_arraylike_not_scalar(init):
            centers = init.toarray() if _is_csr(init) else np.asarray(init)
            return to_table(centers, queue=QM.get_global_queue())
        else:
            raise TypeError("Unsupported type of the `init` value")

        alg = self._get_kmeans_init(
            cluster_count=n_clusters,
            seed=random_seed,
            algorithm=algorithm,
            is_csr=is_csr,
        )
        centers = alg.compute_raw(X_table, dtype, queue=QM.get_global_queue())
        return centers

    def _init_centroids_sklearn(self, X, init, random_state, dtype=None):
        if dtype is None:
            dtype = np.float32

        n_samples = X.shape[0]
        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(X, self.n_clusters, random_state=random_state)
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X[seeds]
        elif callable(init):
            cc_arr = init(X, self.n_clusters, random_state)
            cc_arr = np.ascontiguousarray(cc_arr, dtype=dtype)
            centers = cc_arr
        elif _is_arraylike_not_scalar(init):
            centers = init
        else:
            raise ValueError(
                f"init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{init}' instead."
            )

        return to_table(centers, queue=getattr(QM.get_global_queue(), "_queue", None))

    # --- core train/infer wrappers in the estimator pattern ---

    def _fit_backend(self, X_table, centroids_table, dtype=np.float32, is_csr=False):
        params = self._get_onedal_params(is_csr, dtype)
        result = self.train(params, X_table, centroids_table)
        return (
            result.responses,
            result.objective_function_value,
            result.model,
            result.iteration_count,
        )

    def _predict_backend(self, X_table, result_options=None):
        params = self._get_onedal_params(
            is_csr=False, dtype=X_table.dtype, result_options=result_options
        )
        return self.infer(params, self.model_, X_table)

    # --- public API matched to the pattern ---

    @supports_queue
    def fit(self, X, y=None, queue=None):
        is_csr = _is_csr(X)
        X_table = to_table(X, queue=queue)
        dtype = X_table.dtype

        self._compute_tolerance(X_table, is_csr, dtype)
        self.n_features_in_ = X_table.column_count

        best_model = best_labels = None
        best_inertia = None
        best_n_iter = None

        def is_better(inertia, labels):
            if best_inertia is None:
                return True
            better = inertia < best_inertia
            return better and not self._is_same_clustering(
                labels, best_labels, self.n_clusters
            )

        random_state = check_random_state(self.random_state)

        init = self.init
        use_onedal_init = daal_check_version((2023, "P", 200)) and not callable(self.init)

        # Resolve n_init from 'auto'/'warn' to integer (pattern: like SVM's gamma/max_iter resolution)
        n_init = self.n_init
        default_n_init = 10
        if n_init == "warn":
            warnings.warn(
                (
                    "The default value of `n_init` will change from "
                    f"{default_n_init} to 'auto' in 1.4. Set `n_init` explicitly "
                    "to suppress the warning"
                ),
                FutureWarning,
                stacklevel=2,
            )
            n_init = default_n_init
        if n_init == "auto":
            if isinstance(init, str) and init == "k-means++":
                n_init = 1
            elif isinstance(init, str) and init == "random":
                n_init = default_n_init
            elif callable(init):
                n_init = default_n_init
            else:  # array-like
                n_init = 1

        if _is_arraylike_not_scalar(init) and n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only "
                    f"one init in {self.__class__.__name__} instead of "
                    f"n_init={n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

        for init_idx in range(n_init):
            if use_onedal_init:
                seed = random_state.randint(np.iinfo("i").max)
                centroids_table = self._init_centroids_onedal(
                    X_table, init, seed, is_csr, dtype=dtype
                )
            else:
                centroids_table = self._init_centroids_sklearn(
                    X, init, random_state, dtype=dtype
                )

            if self.verbose:
                print("Initialization complete")

            labels_t, inertia, model, n_iter = self._fit_backend(
                X_table, centroids_table, dtype, is_csr
            )

            if self.verbose:
                print(f"Iteration {n_iter}, inertia {inertia}.")

            if is_better(inertia, labels_t):
                best_model, best_n_iter = model, n_iter
                best_inertia, best_labels = inertia, labels_t

        # assign learned attributes (pattern)
        self.model_ = best_model
        self.n_iter_ = best_n_iter
        self.inertia_ = best_inertia
        self.labels_ = from_table(best_labels).ravel()

        distinct_clusters = len(np.unique(self.labels_))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than n_clusters ({}). "
                "Possibly due to duplicate points in X.".format(
                    distinct_clusters, self.n_clusters
                ),
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    @property
    def cluster_centers_(self):
        if self._cluster_centers_ is None:
            if not hasattr(self, "model_") or self.model_ is None:
                raise NameError("This model has not been trained")
            self._cluster_centers_ = from_table(self.model_.centroids)
        return self._cluster_centers_

    @cluster_centers_.setter
    def cluster_centers_(self, cluster_centers):
        self._cluster_centers_ = np.asarray(cluster_centers)
        self.n_iter_ = 0
        self.inertia_ = 0
        # keep backend model in sync
        self.model_.centroids = to_table(self._cluster_centers_)
        self.n_features_in_ = self.model_.centroids.column_count
        self.labels_ = np.arange(self.model_.centroids.row_count)

    @cluster_centers_.deleter
    def cluster_centers_(self):
        self._cluster_centers_ = None

    @supports_queue
    def predict(self, X, queue=None):
        X_table = to_table(X, queue=QM.get_global_queue())
        result = self._predict_backend(X_table)
        return from_table(result.responses).ravel()

    @supports_queue
    def score(self, X, queue=None):
        X_table = to_table(X, queue=QM.get_global_queue())
        result = self._predict_backend(
            X_table, result_options="compute_exact_objective_function"
        )
        return -1 * result.objective_function_value

    def transform(self, X):
        return euclidean_distances(X, self.cluster_centers_)


class KMeans(_BaseKMeans):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            algorithm=algorithm,
        )
        self.copy_x = copy_x  # stored, but not used by oneDAL path

    @supports_queue
    def fit(self, X, y=None, queue=None):
        return super().fit(X, y=y, queue=queue)

    @supports_queue
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @supports_queue
    def score(self, X, queue=None):
        return super().score(X, queue=queue)


def k_means(
    X,
    n_clusters,
    *,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
    queue=None,
):
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, queue=queue)

    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
