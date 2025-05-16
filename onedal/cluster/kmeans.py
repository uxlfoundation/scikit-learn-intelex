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

import logging
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
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.n_local_trials = n_local_trials

    @bind_default_backend("kmeans_common", no_policy=True)
    def _is_same_clustering(self, labels, best_labels, n_clusters): ...

    @bind_default_backend("kmeans.clustering")
    def train(self, params, X_table, centroids_table): ...

    @bind_default_backend("kmeans.clustering")
    def infer(self, params, model, centroids_table): ...

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _get_kmeans_init(self, cluster_count, seed, algorithm, is_csr):
        return KMeansInit(
            cluster_count=cluster_count,
            seed=seed,
            algorithm=algorithm,
            is_csr=is_csr,
        )

    # Get appropriate backend (required for SPMD)
    def _get_basic_statistics_backend(self, result_options):
        return BasicStatistics(result_options)

    def _tolerance(self, X_table, rtol, is_csr, dtype):
        """Compute absolute tolerance from the relative tolerance"""
        if rtol == 0.0:
            return rtol
        dummy = to_table(None)

        bs = self._get_basic_statistics_backend("variance")

        res = bs._compute_raw(X_table, dummy, dtype, is_csr)
        mean_var = from_table(res["variance"]).mean()

        return mean_var * rtol

    def _check_params_vs_input(
        self, X_table, is_csr, default_n_init=10, dtype=np.float32
    ):
        # n_clusters
        if X_table.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X_table.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = self._tolerance(X_table, self.tol, is_csr, dtype)

        # n-init
        # TODO(1.4): Remove
        self._n_init = self.n_init
        if self._n_init == "warn":
            warnings.warn(
                (
                    "The default value of `n_init` will change from "
                    f"{default_n_init} to 'auto' in 1.4. Set the value of `n_init`"
                    " explicitly to suppress the warning"
                ),
                FutureWarning,
                stacklevel=2,
            )
            self._n_init = default_n_init
        if self._n_init == "auto":
            if isinstance(self.init, str) and self.init == "k-means++":
                self._n_init = 1
            elif isinstance(self.init, str) and self.init == "random":
                self._n_init = default_n_init
            elif callable(self.init):
                self._n_init = default_n_init
            else:  # array-like
                self._n_init = 1

        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.__class__.__name__} instead of "
                    f"n_init={self._n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1
        assert self.algorithm == "lloyd"

    def _get_onedal_params(self, is_csr=False, dtype=np.float32, result_options=None):
        thr = self._tol if hasattr(self, "_tol") else self.tol
        return {
            "fptype": dtype,
            "method": "lloyd_csr" if is_csr else "by_default",
            "seed": -1,
            "max_iteration_count": self.max_iter,
            "cluster_count": self.n_clusters,
            "accuracy_threshold": thr,
            "result_options": "" if result_options is None else result_options,
        }

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
            alg = self._get_kmeans_init(
                cluster_count=n_clusters,
                seed=random_seed,
                algorithm=algorithm,
                is_csr=is_csr,
            )
            # We pass down the queue that was set through the KMeans.fit()
            queue = QM.get_global_queue()
            centers_table = alg.compute_raw(X_table, dtype, queue=queue)
        elif isinstance(init, str) and init == "random":
            algorithm = "random_dense" if not is_csr else "random_csr"
            alg = self._get_kmeans_init(
                cluster_count=n_clusters,
                seed=random_seed,
                algorithm=algorithm,
                is_csr=is_csr,
            )
            # We pass down the queue that was set through the KMeans.fit()
            queue = QM.get_global_queue()
            centers_table = alg.compute_raw(X_table, dtype, queue=queue)
        elif _is_arraylike_not_scalar(init):
            if _is_csr(init):
                # oneDAL KMeans only supports Dense Centroids
                centers = init.toarray()
            else:
                centers = np.asarray(init)
            assert centers.shape[0] == n_clusters
            assert centers.shape[1] == X_table.column_count
            # KMeans is implemented on both CPU and GPU for Dense and CSR data
            # The original policy can be used here
            centers_table = to_table(centers, queue=QM.get_global_queue())
        else:
            raise TypeError("Unsupported type of the `init` value")

        return centers_table

    def _init_centroids_sklearn(self, X, init, random_state, dtype=np.float32):
        # For oneDAL versions < 2023.2 or callable init,
        # using the scikit-learn implementation
        logging.getLogger("sklearnex").info("Computing KMeansInit with Stock sklearn")
        n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                self.n_clusters,
                random_state=random_state,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X[seeds]
        elif callable(init):
            cc_arr = init(X, self.n_clusters, random_state)
            cc_arr = np.ascontiguousarray(cc_arr, dtype=dtype)
            self._validate_center_shape(X, cc_arr)
            centers = cc_arr
        elif _is_arraylike_not_scalar(init):
            centers = init
        else:
            raise ValueError(
                f"init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{ init }' instead."
            )

        return to_table(centers, queue=getattr(QM.get_global_queue(), "_queue", None))

    def _fit_backend(self, X_table, centroids_table, dtype=np.float32, is_csr=False):
        params = self._get_onedal_params(is_csr, dtype)

        assert X_table.dtype == dtype

        result = self.train(params, X_table, centroids_table)

        return (
            result.responses,
            result.objective_function_value,
            result.model,
            result.iteration_count,
        )

    def _fit(self, X):
        is_csr = _is_csr(X)

        if _get_config()["use_raw_input"] is False:
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                accept_sparse="csr",
                force_all_finite=False,
            )
        X_table = to_table(X, queue=QM.get_global_queue())
        dtype = X_table.dtype

        self._check_params_vs_input(X_table, is_csr, dtype=dtype)

        self.n_features_in_ = X_table.column_count

        best_model, best_n_iter = None, None
        best_inertia, best_labels = None, None

        def is_better_iteration(inertia, labels):
            if best_inertia is None:
                return True
            else:
                better_inertia = inertia < best_inertia
                return better_inertia and not self._is_same_clustering(
                    labels, best_labels, self.n_clusters
                )

        random_state = check_random_state(self.random_state)

        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = _check_array(
                init, dtype=dtype, accept_sparse="csr", copy=True, order="C"
            )
            self._validate_center_shape(X, init)

        use_onedal_init = daal_check_version((2023, "P", 200)) and not callable(self.init)

        for _ in range(self._n_init):
            if use_onedal_init:
                random_seed = random_state.randint(np.iinfo("i").max)
                centroids_table = self._init_centroids_onedal(
                    X_table, init, random_seed, is_csr, dtype=dtype
                )
            else:
                centroids_table = self._init_centroids_sklearn(
                    X, init, random_state, dtype=dtype
                )

            if self.verbose:
                print("Initialization complete")

            labels, inertia, model, n_iter = self._fit_backend(
                X_table, centroids_table, dtype, is_csr
            )

            if self.verbose:
                print("Iteration {}, inertia {}.".format(n_iter, inertia))

            if is_better_iteration(inertia, labels):
                best_model, best_n_iter = model, n_iter
                best_inertia, best_labels = inertia, labels

        # Types without conversion
        self.model_ = best_model

        # Simple types
        self.n_iter_ = best_n_iter
        self.inertia_ = best_inertia

        # Complex type conversion
        self.labels_ = from_table(best_labels).ravel()

        distinct_clusters = len(np.unique(self.labels_))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        return self

    @property
    def cluster_centers_(self):
        if not hasattr(self, "_cluster_centers_"):
            if hasattr(self, "model_"):
                centroids = self.model_.centroids
                self._cluster_centers_ = from_table(centroids)
            else:
                raise NameError("This model has not been trained")
        return self._cluster_centers_

    @cluster_centers_.setter
    def cluster_centers_(self, cluster_centers):
        self._cluster_centers_ = np.asarray(cluster_centers)

        self.n_iter_ = 0
        self.inertia_ = 0

        self.model_.centroids = to_table(self._cluster_centers_)
        self.n_features_in_ = self.model_.centroids.column_count
        self.labels_ = np.arange(self.model_.centroids.row_count)

        return self

    @cluster_centers_.deleter
    def cluster_centers_(self):
        del self._cluster_centers_

    def _predict(self, X, result_options=None):
        is_csr = _is_csr(X)

        X_table = to_table(X, queue=QM.get_global_queue())
        params = self._get_onedal_params(is_csr, X_table.dtype, result_options)

        result = self.infer(params, self.model_, X_table)

        if result_options == "compute_exact_objective_function":
            # This is only set for score function
            return -1 * result.objective_function_value
        else:
            return from_table(result.responses).ravel()

    def _score(self, X):
        result_options = "compute_exact_objective_function"

        return self._predict(
            X,
            result_options,
        )

    def _transform(self, X):
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
        )

        self.copy_x = copy_x
        self.algorithm = algorithm
        assert self.algorithm == "lloyd"

    @supports_queue
    def fit(self, X, y=None, queue=None):
        return self._fit(X)

    @supports_queue
    def predict(self, X, queue=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self._predict(X)

    def fit_predict(self, X, y=None, queue=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, queue=queue).labels_

    def fit_transform(self, X, y=None, queue=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, queue=queue)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """

        return self._transform(X)

    @supports_queue
    def score(self, X, queue=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        score: float
            Opposite of the value of X on the K-means objective.
        """
        return self._score(X)


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
