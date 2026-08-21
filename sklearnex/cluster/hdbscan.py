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

from daal4py.sklearn._utils import (
    daal_check_build_date,
    daal_check_version,
    sklearn_check_version,
)

# HDBSCAN was added to oneDAL in 2026.2, mid-cycle, hence the build date check. The
# whole module is gated, rather than just the 'onedal' import, so that it stays
# importable with an older oneDAL: 'sklearn.utils.all_estimators' imports every
# sklearnex module directly, so a module-level 'raise ImportError' or an unguarded
# 'onedal.cluster' import would break the estimator discovery that
# 'sklearnex/tests/test_common.py' relies on.
if daal_check_version((2026, "P", 200)) and daal_check_build_date(20260814):
    import warnings
    from functools import partial

    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
    from sklearn.utils.validation import _num_samples, check_array

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import is_sparse
    from onedal.cluster import HDBSCAN as onedal_HDBSCAN

    from .._device_offload import dispatch
    from .._utils import PatchingConditionsChain
    from ..base import oneDALEstimator
    from ..utils._array_api import enable_array_api, get_namespace
    from ..utils.validation import assert_all_finite, validate_data

    # lax conversion of the input for the finiteness check done while
    # determining whether oneDAL can be used, the actual validation of
    # the data happens in '_onedal_fit'
    _check_array = partial(
        check_array,
        dtype=None,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_min_features=0,
        accept_sparse=False,
        ensure_all_finite=False,
    )

    @enable_array_api
    @control_n_jobs(decorated_methods=["fit"])
    class HDBSCAN(oneDALEstimator, _sklearn_HDBSCAN):
        __doc__ = _sklearn_HDBSCAN.__doc__

        # copied to keep 'control_n_jobs' from modifying scikit-learn's own constraints
        _parameter_constraints: dict = {**_sklearn_HDBSCAN._parameter_constraints}

        # scikit-learn's '__init__' is used as-is, all of its parameters are
        # forwarded to the onedal estimator or checked for oneDAL support

        _onedal_hdbscan = staticmethod(onedal_HDBSCAN)

        def _onedal_method(self) -> str:
            """Translate scikit-learn's 'algorithm' into the oneDAL method."""
            if self.algorithm == "auto":
                # the kd-tree based neighbors search is the fastest option, but
                # oneDAL only implements it for a subset of the distances
                return (
                    "kd_tree" if self.metric in self._kd_tree_metrics else "brute_force"
                )
            if self.algorithm == "brute":
                return "brute_force"
            # 'kd_tree' and 'ball_tree' are named the same way in oneDAL
            return self.algorithm

        def _onedal_fit(self, X, queue=None):
            # oneDAL never writes into the data, so 'copy' has no effect here, but the
            # deprecation of its default has to be repeated for the offloaded path
            # TODO(sklearn 1.10): remove, the parameter loses its "warn" default
            if (
                sklearn_check_version("1.8")
                and not sklearn_check_version("1.10")
                and self.copy == "warn"
            ):
                warnings.warn(
                    "The default value of `copy` will change from False to True in 1.10."
                    " Explicitly set a value for `copy` to silence this warning.",
                    FutureWarning,
                )

            xp, _ = get_namespace(X)
            X = validate_data(
                self, X, accept_sparse=False, dtype=[xp.float64, xp.float32]
            )

            metric_params = self.metric_params or {}
            onedal_params = {
                # sklearn takes 'min_cluster_size' as 'min_samples' when unset
                "min_cluster_size": self.min_cluster_size,
                "min_samples": (
                    self.min_cluster_size
                    if self.min_samples is None
                    else self.min_samples
                ),
                "metric": self.metric,
                "degree": metric_params.get("p", 2.0),
                "alpha": self.alpha,
                "method": self._onedal_method(),
                "leaf_size": self.leaf_size,
                "cluster_selection": self.cluster_selection_method,
                "allow_single_cluster": self.allow_single_cluster,
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
                # oneDAL takes zero as 'no limit on the size of a cluster'
                "max_cluster_size": self.max_cluster_size or 0,
                "store_centers": self.store_centers or "none",
            }
            self._onedal_estimator = self._onedal_hdbscan(**onedal_params)

            self._onedal_estimator.fit(X, queue=queue)
            self.labels_ = self._onedal_estimator.labels_
            self.n_features_in_ = X.shape[1]

            # oneDAL leaves the centers out when it does not find any cluster,
            # while scikit-learn returns them empty
            if self.store_centers in ("centroid", "both"):
                self.centroids_ = getattr(self._onedal_estimator, "centroids_", X[:0, :])
            if self.store_centers in ("medoid", "both"):
                self.medoids_ = getattr(self._onedal_estimator, "medoids_", X[:0, :])

            # scikit-learn derives the membership strengths from the lambda values
            # of the condensed tree, which oneDAL does not return, so the degree to
            # which a sample persists in its cluster is unknown. Noise is reported
            # as zero, as scikit-learn does, and the members of a cluster as one
            self.probabilities_ = xp.astype(self.labels_ != -1, xp.float64)

        # metrics as named by scikit-learn, mapping onto oneDAL's distances
        # happens in '_onedal_fit'
        _onedal_supported_metrics = (
            "euclidean",
            "manhattan",
            "minkowski",
            "chebyshev",
            "cosine",
        )
        # distances for which oneDAL implements a kd-tree based neighbors search
        _kd_tree_metrics = ("euclidean", "manhattan", "minkowski", "chebyshev")
        # oneDAL computes the cosine distance only in its brute force method
        _cosine_algorithms = ("auto", "brute")

        def _onedal_supported(self, method_name, *data):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.cluster.{class_name}.{method_name}"
            )
            if method_name == "fit":
                X = data[0]
                # sklearn takes 'min_cluster_size' as 'min_samples' when unset
                min_samples = (
                    self.min_cluster_size
                    if self.min_samples is None
                    else self.min_samples
                )
                dal_ready = patching_status.and_conditions(
                    [
                        (
                            self.metric in self._onedal_supported_metrics,
                            f"'{self.metric}' metric is not supported. Only 'euclidean', "
                            "'manhattan', 'minkowski', 'chebyshev' and 'cosine' are "
                            "supported.",
                        ),
                        (
                            self.metric != "cosine"
                            or self.algorithm in self._cosine_algorithms,
                            "'cosine' metric is only supported by the 'auto' and 'brute' "
                            "algorithms.",
                        ),
                        (not is_sparse(X), "X is sparse. Sparse input is not supported."),
                        (
                            min_samples <= _num_samples(X),
                            "min_samples is larger than the number of samples in X.",
                        ),
                    ]
                )
                if not dal_ready:
                    return patching_status

                # sklearn labels non-finite samples as special outliers, while
                # oneDAL does not support them
                try:
                    assert_all_finite(_check_array(X))
                except ValueError:
                    patching_status.and_conditions(
                        [(False, "Missing values and infinites are not supported.")]
                    )
                return patching_status
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def fit(self, X, y=None):
            self._validate_params()

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": _sklearn_HDBSCAN.fit,
                },
                X,
            )

            return self

        fit.__doc__ = _sklearn_HDBSCAN.fit.__doc__
