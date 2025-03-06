# ==============================================================================
# Copyright 2014 Intel Corporation
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

import numbers
import warnings

import numpy as np
from scipy import sparse as sp
from sklearn.cluster import KMeans as KMeans_original
from sklearn.cluster._kmeans import _labels_inertia
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import (
    _deprecate_positional_args,
    _num_samples,
    check_is_fitted,
)

import daal4py

from .._n_jobs_support import control_n_jobs
from .._utils import PatchingConditionsChain, getFPType, sklearn_check_version
from ..utils.validation import check_feature_names, validate_data

if sklearn_check_version("1.1"):
    from sklearn.utils.validation import _check_sample_weight, _is_arraylike_not_scalar


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if centers.shape[0] != n_centers:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of clusters {n_centers}."
        )
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of features of the data {X.shape[1]}."
        )


def _tolerance(X, rtol):
    """Compute absolute tolerance from the relative tolerance"""
    if rtol == 0.0:
        return rtol
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
        mean_var = np.mean(variances)
    else:
        mean_var = np.var(X, axis=0).mean()
    return mean_var * rtol


def _daal4py_compute_starting_centroids(
    X, X_fptype, nClusters, cluster_centers_0, verbose, random_state
):
    def is_string(s, target_str):
        return isinstance(s, str) and s == target_str

    is_sparse = sp.issparse(X)

    deterministic = False
    if is_string(cluster_centers_0, "k-means++"):
        _seed = random_state.randint(np.iinfo("i").max)
        plus_plus_method = "plusPlusCSR" if is_sparse else "plusPlusDense"
        daal_engine = daal4py.engines_mt19937(
            fptype=X_fptype, method="defaultDense", seed=_seed
        )
        _n_local_trials = 2 + int(np.log(nClusters))
        kmeans_init = daal4py.kmeans_init(
            nClusters,
            fptype=X_fptype,
            nTrials=_n_local_trials,
            method=plus_plus_method,
            engine=daal_engine,
        )
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    elif is_string(cluster_centers_0, "random"):
        _seed = random_state.randint(np.iinfo("i").max)
        random_method = "randomCSR" if is_sparse else "randomDense"
        daal_engine = daal4py.engines_mt19937(
            seed=_seed, fptype=X_fptype, method="defaultDense"
        )
        kmeans_init = daal4py.kmeans_init(
            nClusters,
            fptype=X_fptype,
            method=random_method,
            engine=daal_engine,
        )
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    elif hasattr(cluster_centers_0, "__array__"):
        deterministic = True
        cc_arr = np.ascontiguousarray(cluster_centers_0, dtype=X.dtype)
        _validate_center_shape(X, nClusters, cc_arr)
        centroids_ = cc_arr
    elif callable(cluster_centers_0):
        cc_arr = cluster_centers_0(X, nClusters, random_state)
        cc_arr = np.ascontiguousarray(cc_arr, dtype=X.dtype)
        _validate_center_shape(X, nClusters, cc_arr)
        centroids_ = cc_arr
    elif is_string(cluster_centers_0, "deterministic"):
        deterministic = True
        default_method = "lloydCSR" if is_sparse else "defaultDense"
        kmeans_init = daal4py.kmeans_init(
            nClusters, fptype=X_fptype, method=default_method
        )
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    else:
        raise ValueError(
            f"init should be either 'k-means++', 'random', a ndarray or a "
            f"callable, got '{cluster_centers_0}' instead."
        )
    if verbose:
        print("Initialization complete")
    return deterministic, centroids_


def _daal4py_kmeans_compatibility(
    nClusters,
    maxIterations,
    fptype="double",
    method="lloydDense",
    accuracyThreshold=0.0,
    resultsToEvaluate="computeCentroids",
    gamma=1.0,
):
    kmeans_algo = daal4py.kmeans(
        nClusters=nClusters,
        maxIterations=maxIterations,
        fptype=fptype,
        resultsToEvaluate=resultsToEvaluate,
        accuracyThreshold=accuracyThreshold,
        method=method,
        gamma=gamma,
    )
    return kmeans_algo


def _daal4py_k_means_predict(
    X, nClusters, centroids, resultsToEvaluate="computeAssignments"
):
    X_fptype = getFPType(X)
    is_sparse = sp.issparse(X)
    method = "lloydCSR" if is_sparse else "defaultDense"
    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters=nClusters,
        maxIterations=0,
        fptype=X_fptype,
        resultsToEvaluate=resultsToEvaluate,
        method=method,
    )

    res = kmeans_algo.compute(X, centroids)

    return res.assignments[:, 0], res.objectiveFunction[0, 0]


def _daal4py_k_means_fit(
    X, nClusters, numIterations, tol, cluster_centers_0, n_init, verbose, random_state
):
    if numIterations < 0:
        raise ValueError("Wrong iterations number")

    def is_string(s, target_str):
        return isinstance(s, str) and s == target_str

    default_n_init = 10
    if n_init in ["auto", "warn"]:
        if n_init == "warn" and sklearn_check_version("1.2"):
            warnings.warn(
                "The default value of `n_init` will change from "
                f"{default_n_init} to 'auto' in 1.4. Set the value of `n_init`"
                " explicitly to suppress the warning",
                FutureWarning,
            )
        if is_string(cluster_centers_0, "k-means++"):
            n_init = 1
        else:
            n_init = default_n_init
    X_fptype = getFPType(X)
    abs_tol = _tolerance(X, tol)  # tol is relative tolerance
    is_sparse = sp.issparse(X)
    method = "lloydCSR" if is_sparse else "defaultDense"
    best_inertia, best_cluster_centers = None, None
    best_n_iter = -1
    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters=nClusters,
        maxIterations=numIterations,
        accuracyThreshold=abs_tol,
        fptype=X_fptype,
        resultsToEvaluate="computeCentroids",
        method=method,
    )

    for k in range(n_init):
        deterministic, starting_centroids_ = _daal4py_compute_starting_centroids(
            X, X_fptype, nClusters, cluster_centers_0, verbose, random_state
        )

        res = kmeans_algo.compute(X, starting_centroids_)

        inertia = res.objectiveFunction[0, 0]
        if verbose:
            print(f"Iteration {k}, inertia {inertia}.")

        if best_inertia is None or inertia < best_inertia:
            best_cluster_centers = res.centroids
            if n_init > 1:
                best_cluster_centers = best_cluster_centers.copy()
            best_inertia = inertia
            best_n_iter = int(res.nIterations[0, 0])
        if deterministic and n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d" % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            break

    flag_compute = "computeAssignments|computeExactObjectiveFunction"
    best_labels, best_inertia = _daal4py_k_means_predict(
        X, nClusters, best_cluster_centers, flag_compute
    )

    distinct_clusters = np.unique(best_labels).size
    if distinct_clusters < nClusters:
        warnings.warn(
            "Number of distinct clusters ({}) found smaller than "
            "n_clusters ({}). Possibly due to duplicate points "
            "in X.".format(distinct_clusters, nClusters),
            ConvergenceWarning,
            stacklevel=2,
        )
        # for passing test case "test_kmeans_warns_less_centers_than_unique_points"

    return best_cluster_centers, best_labels, best_inertia, best_n_iter


def _fit(self, X, y=None, sample_weight=None):
    init = self.init
    if sklearn_check_version("1.1"):
        if sklearn_check_version("1.2"):
            self._validate_params()

        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        if sklearn_check_version("1.2"):
            self._check_params_vs_input(X)
        else:
            self._check_params(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)
    else:
        if hasattr(self, "precompute_distances"):
            if self.precompute_distances != "deprecated":
                warnings.warn(
                    "'precompute_distances' was deprecated in version "
                    "0.23 and will be removed in 1.0 (renaming of 0.25)."
                    " It has no effect",
                    FutureWarning,
                )

        self._n_threads = None
        if hasattr(self, "n_jobs"):
            if self.n_jobs != "deprecated":
                warnings.warn(
                    "'n_jobs' was deprecated in version 0.23 and will be"
                    " removed in 1.0 (renaming of 0.25).",
                    FutureWarning,
                )
                self._n_threads = self.n_jobs
        self._n_threads = _openmp_effective_n_threads(self._n_threads)

        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")

        random_state = check_random_state(self.random_state)
        check_feature_names(self, X, reset=True)

        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        algorithm = self.algorithm
        if sklearn_check_version("1.2"):
            if algorithm == "elkan" and self.n_clusters == 1:
                warnings.warn(
                    "algorithm='elkan' doesn't make sense for a single "
                    "cluster. Using 'full' instead.",
                    RuntimeWarning,
                )
                algorithm = "lloyd"

            if algorithm == "auto" or algorithm == "full":
                warnings.warn(
                    "algorithm= {'auto','full'} is deprecated" "Using 'lloyd' instead.",
                    RuntimeWarning,
                )
                algorithm = "lloyd" if self.n_clusters == 1 else "elkan"

            if algorithm not in ["lloyd", "full", "elkan"]:
                raise ValueError(
                    "Algorithm must be 'auto','lloyd', 'full' or 'elkan',"
                    "got {}".format(str(algorithm))
                )
        else:
            if algorithm == "elkan" and self.n_clusters == 1:
                warnings.warn(
                    "algorithm='elkan' doesn't make sense for a single "
                    "cluster. Using 'full' instead.",
                    RuntimeWarning,
                )
                algorithm = "full"

            if algorithm == "auto":
                algorithm = "full" if self.n_clusters == 1 else "elkan"

            if algorithm not in ["full", "elkan"]:
                raise ValueError(
                    "Algorithm must be 'auto', 'full' or 'elkan', got"
                    " {}".format(str(algorithm))
                )

    X_len = _num_samples(X)

    _patching_status = PatchingConditionsChain("sklearn.cluster.KMeans.fit")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                self.n_clusters <= X_len,
                "The number of clusters is larger than the number of samples in X.",
            )
        ]
    )

    if _dal_ready and sample_weight is not None:
        if isinstance(sample_weight, numbers.Number):
            sample_weight = np.full(X_len, sample_weight, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight)
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    sample_weight.shape == (X_len,),
                    "Sample weights do not have the same length as X.",
                ),
                (
                    np.allclose(sample_weight, np.ones_like(sample_weight)),
                    "Sample weights are not ones.",
                ),
            ]
        )

    _patching_status.write_log()
    if _dal_ready:
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        self.n_features_in_ = X.shape[1]
        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
        ) = _daal4py_k_means_fit(
            X,
            self.n_clusters,
            self.max_iter,
            self.tol,
            init,
            self.n_init,
            self.verbose,
            random_state,
        )
        if sklearn_check_version("1.1"):
            self._n_features_out = self.cluster_centers_.shape[0]
    else:
        super(KMeans, self).fit(X, y=y, sample_weight=sample_weight)
    return self


def _daal4py_check_test_data(self, X):
    check_feature_names(self, X, reset=False)
    X = check_array(
        X, accept_sparse="csr", dtype=[np.float64, np.float32], accept_large_sparse=False
    )
    if self.n_features_in_ != X.shape[1]:
        raise ValueError(
            (
                f"X has {X.shape[1]} features, "
                f"but Kmeans is expecting {self.n_features_in_} features as input"
            )
        )
    return X


def _predict(self, X, sample_weight=None):
    check_is_fitted(self)

    X = _daal4py_check_test_data(self, X)

    if (
        sklearn_check_version("1.3")
        and isinstance(sample_weight, str)
        and sample_weight == "deprecated"
    ):
        sample_weight = None

    _patching_status = PatchingConditionsChain("sklearn.cluster.KMeans.predict")
    _patching_status.and_conditions(
        [
            (sample_weight is None, "Sample weights are not supported."),
            (hasattr(X, "__array__"), "X does not have '__array__' attribute."),
        ]
    )

    # CSR array is introduced in scipy 1.11, this requires an initial attribute check
    if hasattr(sp, "csr_array"):
        _dal_ready = _patching_status.or_conditions(
            [
                (
                    sp.isspmatrix_csr(X) or isinstance(X, sp.csr_array),
                    "X is not csr sparse.",
                )
            ]
        )
    else:
        _dal_ready = _patching_status.or_conditions(
            [(sp.isspmatrix_csr(X), "X is not csr sparse.")]
        )

    _patching_status.write_log()
    if _dal_ready:
        return _daal4py_k_means_predict(X, self.n_clusters, self.cluster_centers_)[0]
    if sklearn_check_version("1.2"):
        if sklearn_check_version("1.3") and sample_weight is not None:
            warnings.warn(
                "'sample_weight' was deprecated in version 1.3 and "
                "will be removed in 1.5.",
                FutureWarning,
            )
        return _labels_inertia(X, sample_weight, self.cluster_centers_)[0]
    else:
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, sample_weight, x_squared_norms, self.cluster_centers_)[
            0
        ]


@control_n_jobs(decorated_methods=["fit", "predict"])
class KMeans(KMeans_original):
    __doc__ = KMeans_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**KMeans_original._parameter_constraints}

        @_deprecate_positional_args
        def __init__(
            self,
            n_clusters=8,
            *,
            init="k-means++",
            n_init="auto" if sklearn_check_version("1.4") else "warn",
            max_iter=300,
            tol=1e-4,
            verbose=0,
            random_state=None,
            copy_x=True,
            algorithm="lloyd",
        ):
            super(KMeans, self).__init__(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                verbose=verbose,
                random_state=random_state,
                copy_x=copy_x,
                algorithm=algorithm,
            )

    else:

        @_deprecate_positional_args
        def __init__(
            self,
            n_clusters=8,
            *,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-4,
            verbose=0,
            random_state=None,
            copy_x=True,
            algorithm="lloyd" if sklearn_check_version("1.1") else "auto",
        ):
            super(KMeans, self).__init__(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                verbose=verbose,
                random_state=random_state,
                copy_x=copy_x,
                algorithm=algorithm,
            )

    def fit(self, X, y=None, sample_weight=None):
        return _fit(self, X, y=y, sample_weight=sample_weight)

    if sklearn_check_version("1.5"):

        def predict(self, X):
            return _predict(self, X)

    else:

        def predict(
            self, X, sample_weight="deprecated" if sklearn_check_version("1.3") else None
        ):
            return _predict(self, X, sample_weight=sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        return super().fit_predict(X, y, sample_weight)

    fit.__doc__ = KMeans_original.fit.__doc__
    predict.__doc__ = KMeans_original.predict.__doc__
    fit_predict.__doc__ = KMeans_original.fit_predict.__doc__
