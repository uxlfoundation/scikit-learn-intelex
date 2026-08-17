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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

if sklearn_check_version("1.3") and daal_check_version((2026, "P", 200)):

    from functools import partial

    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
    from sklearn.utils.validation import _num_samples, check_array

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import is_sparse
    from onedal.cluster.hdbscan import HDBSCAN as onedal_HDBSCAN
    from onedal.utils._array_api import _is_numpy_namespace

    from .._device_offload import dispatch
    from .._utils import PatchingConditionsChain
    from ..base import oneDALEstimator
    from ..utils._array_api import enable_array_api, get_namespace
    from ..utils.validation import assert_all_finite, validate_data

    if sklearn_check_version("1.9"):
        from sklearn.utils._array_api import get_namespace_and_device

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

        _parameter_constraints: dict = {**_sklearn_HDBSCAN._parameter_constraints}

        def __init__(
            self,
            min_cluster_size=5,
            *,
            min_samples=None,
            cluster_selection_epsilon=0.0,
            max_cluster_size=None,
            metric="euclidean",
            metric_params=None,
            alpha=1.0,
            algorithm="auto",
            leaf_size=40,
            n_jobs=None,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            store_centers=None,
            copy="warn" if sklearn_check_version("1.8") else False,
        ):
            super().__init__(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                max_cluster_size=max_cluster_size,
                metric=metric,
                metric_params=metric_params,
                alpha=alpha,
                algorithm=algorithm,
                leaf_size=leaf_size,
                n_jobs=n_jobs,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster,
                store_centers=store_centers,
                copy=copy,
            )

        _onedal_hdbscan = staticmethod(onedal_HDBSCAN)

        def _onedal_fit(self, X, y=None, queue=None):
            if sklearn_check_version("1.9"):
                xp, _, device = get_namespace_and_device(X)
            else:
                xp, _ = get_namespace(X)
                device = getattr(X, "device", None)
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
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
                "max_cluster_size": self.max_cluster_size,
                "store_centers": self.store_centers,
                "copy": self.copy,
            }
            self._onedal_estimator = self._onedal_hdbscan(**onedal_params)

            self._onedal_estimator.fit(X, queue=queue)
            self.labels_ = self._onedal_estimator.labels_
            self.n_features_in_ = X.shape[1]

            if hasattr(self._onedal_estimator, "centroids_"):
                self.centroids_ = self._onedal_estimator.centroids_
            if hasattr(self._onedal_estimator, "medoids_"):
                self.medoids_ = self._onedal_estimator.medoids_

            # oneDAL does not compute probabilities; set uniform zeros
            # to match sklearn's interface
            kwargs = {"dtype": xp.float64}  # always the same
            if not _is_numpy_namespace(xp):
                kwargs["device"] = device
            self.probabilities_ = xp.zeros(self.labels_.shape[0], **kwargs)

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
                            f"'{self.metric}' metric is not supported. "
                            f"Only {self._onedal_supported_metrics} are supported.",
                        ),
                        (
                            self.metric != "cosine"
                            or self.algorithm in ("auto", "brute", "brute_force"),
                            "'cosine' metric is only supported by the 'auto' and "
                            "'brute' algorithms.",
                        ),
                        (
                            self.cluster_selection_method in ("eom", "leaf"),
                            f"'{self.cluster_selection_method}' cluster selection "
                            "method is not supported. Only 'eom' and 'leaf' are supported.",
                        ),
                        (
                            self.algorithm
                            in (
                                "auto",
                                "brute",
                                "brute_force",
                                "kd_tree",
                                "kdtree",
                                "ball_tree",
                                "balltree",
                            ),
                            f"'{self.algorithm}' algorithm is not supported. "
                            "Only 'auto', 'brute', 'kd_tree', and 'ball_tree' are supported.",
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
                y,
            )

            return self

        fit.__doc__ = _sklearn_HDBSCAN.fit.__doc__
