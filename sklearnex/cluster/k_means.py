# ==============================================================================
# Copyright 2021 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, "P", 200)):

    import numbers
    import warnings

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.cluster import KMeans as _sklearn_KMeans
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from sklearn.utils.validation import (
        _check_sample_weight,
        _num_samples,
        check_is_fitted,
    )

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import sklearn_check_version
    from onedal._device_offload import support_input_format
    from onedal.cluster import KMeans as onedal_KMeans
    from onedal.utils.validation import _is_csr

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain
    from ..base import oneDALEstimator
    from ..utils._array_api import get_namespace
    from ..utils.validation import validate_data

    @control_n_jobs(decorated_methods=["fit", "fit_transform", "predict", "score"])
    class KMeans(oneDALEstimator, _sklearn_KMeans):
        __doc__ = _sklearn_KMeans.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**_sklearn_KMeans._parameter_constraints}

        def __init__(
            self,
            n_clusters=8,
            *,
            init="k-means++",
            n_init=(
                "auto"
                if sklearn_check_version("1.4")
                else "warn" if sklearn_check_version("1.2") else 10
            ),
            max_iter=300,
            tol=1e-4,
            verbose=0,
            random_state=None,
            copy_x=True,
            algorithm="lloyd" if sklearn_check_version("1.1") else "auto",
        ):
            super().__init__(
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

        def _initialize_onedal_estimator(self):
            onedal_params = {
                "n_clusters": self.n_clusters,
                "init": self.init,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "n_init": self.n_init,
                "verbose": self.verbose,
                "random_state": self.random_state,
            }

            self._onedal_estimator = onedal_KMeans(**onedal_params)

        def _onedal_fit_supported(self, method_name, X, y=None, sample_weight=None):
            assert method_name == "fit"

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(f"sklearn.cluster.{class_name}.fit")

            sample_count = _num_samples(X)
            supported_algs = ["auto", "full", "lloyd", "elkan"]

            if self.algorithm == "elkan":
                logging.getLogger("sklearnex").info(
                    "oneDAL does not support 'elkan', using 'lloyd' algorithm instead."
                )
            correct_count = self.n_clusters < sample_count

            is_data_supported = (
                _is_csr(X) and daal_check_version((2024, "P", 700))
            ) or not issparse(X)

            _acceptable_sample_weights = self._validate_sample_weight(sample_weight, X)

            patching_status.and_conditions(
                [
                    (
                        self.algorithm in supported_algs,
                        "Only 'lloyd' algorithm is supported, 'elkan' is computed using lloyd",
                    ),
                    (self.n_clusters != 1, "n_clusters=1 is not supported"),
                    (correct_count, "n_clusters is smaller than number of samples"),
                    (
                        _acceptable_sample_weights,
                        "oneDAL doesn't support sample_weight. Accepted options are None, constant, or equal weights.",
                    ),
                    (
                        is_data_supported,
                        "Supported data formats: Dense, CSR (oneDAL version >= 2024.7.0).",
                    ),
                ]
            )

            return patching_status

        def fit(self, X, y=None, sample_weight=None):
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": _sklearn_KMeans.fit,
                },
                X,
                y,
                sample_weight,
            )

            return self

        def _onedal_fit(self, X, _, sample_weight, queue=None):
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

            self._n_features_out = self.n_clusters

            self._initialize_onedal_estimator()
            self._n_threads = _openmp_effective_n_threads()
            self._onedal_estimator.fit(X, queue=queue)

            self._save_attributes()

        def _validate_sample_weight(self, sample_weight, X):
            if sample_weight is None:
                return True
            elif isinstance(sample_weight, numbers.Number) or isinstance(
                sample_weight, str
            ):
                return True
            else:
                sample_weight = _check_sample_weight(
                    sample_weight,
                    X,
                    dtype=X.dtype if hasattr(X, "dtype") else None,
                )
                if np.all(sample_weight == sample_weight[0]):
                    return True
                else:
                    return False

        def _onedal_predict_supported(self, method_name, *data):
            class_name = self.__class__.__name__

            patching_status = PatchingConditionsChain(
                f"sklearn.cluster.{class_name}.{method_name}"
            )

            X = data[0]
            sample_weight = data[-1] if len(data) > 1 else None

            is_data_supported = (
                _is_csr(X) and daal_check_version((2024, "P", 700))
            ) or not issparse(X)

            # algorithm "auto" has been deprecated since 1.1,
            # algorithm "full" has been replaced by "lloyd"
            supported_algs = ["auto", "full", "lloyd", "elkan"]
            if self.algorithm == "elkan":
                logging.getLogger("sklearnex").info(
                    "oneDAL does not support 'elkan', using 'lloyd' algorithm instead."
                )

            _acceptable_sample_weights = True
            if not sklearn_check_version("1.5"):
                _acceptable_sample_weights = self._validate_sample_weight(
                    sample_weight, X
                )

            patching_status.and_conditions(
                [
                    (
                        self.algorithm in supported_algs,
                        "Only 'lloyd' algorithm is supported, 'elkan' is computed using lloyd.",
                    ),
                    (self.n_clusters != 1, "n_clusters=1 is not supported"),
                    (
                        is_data_supported,
                        "Supported data formats: Dense, CSR (oneDAL version >= 2024.7.0).",
                    ),
                    (
                        _acceptable_sample_weights,
                        "oneDAL doesn't support sample_weight. Acceptable options are None, constant, or equal weights.",
                    ),
                ]
            )

            return patching_status

        if sklearn_check_version("1.5"):

            @wrap_output_data
            def predict(self, X):
                self._validate_params()
                check_is_fitted(self)
                return dispatch(
                    self,
                    "predict",
                    {
                        "onedal": self.__class__._onedal_predict,
                        "sklearn": _sklearn_KMeans.predict,
                    },
                    X,
                )

        else:

            @wrap_output_data
            def predict(
                self,
                X,
                sample_weight="deprecated" if sklearn_check_version("1.3") else None,
            ):
                if sklearn_check_version("1.2"):
                    self._validate_params()

                if sklearn_check_version("1.3"):
                    if isinstance(sample_weight, str) and sample_weight == "deprecated":
                        sample_weight = None

                    if sample_weight is not None:
                        warnings.warn(
                            "'sample_weight' was deprecated in version 1.3 and "
                            "will be removed in 1.5.",
                            FutureWarning,
                        )
                check_is_fitted(self)
                return dispatch(
                    self,
                    "predict",
                    {
                        "onedal": self.__class__._onedal_predict,
                        "sklearn": _sklearn_KMeans.predict,
                    },
                    X,
                    sample_weight,
                )

        def _onedal_predict(self, X, sample_weight=None, queue=None):
            X = validate_data(
                self,
                X,
                accept_sparse="csr",
                reset=False,
                dtype=[np.float64, np.float32],
            )

            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.cluster_centers_ = self.cluster_centers_

            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_supported(self, method_name, *data):
            if method_name == "fit":
                return self._onedal_fit_supported(method_name, *data)
            if method_name in ["predict", "score"]:
                return self._onedal_predict_supported(method_name, *data)
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        _onedal_gpu_supported = _onedal_supported
        _onedal_cpu_supported = _onedal_supported

        def _check_test_data(self, X):
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                accept_sparse="csr",
                reset=False,
                dtype=[xp.float64, xp.float32],
                order="C",
                accept_large_sparse=False,
            )
            return X

        _transform = support_input_format(_sklearn_KMeans._transform)

        @wrap_output_data
        def score(self, X, y=None, sample_weight=None):
            check_is_fitted(self)
            return dispatch(
                self,
                "score",
                {
                    "onedal": self.__class__._onedal_score,
                    "sklearn": _sklearn_KMeans.score,
                },
                X,
                y,
                sample_weight,
            )

        def _onedal_score(self, X, y=None, sample_weight=None, queue=None):
            X = validate_data(
                self,
                X,
                accept_sparse="csr",
                reset=False,
                dtype=[np.float64, np.float32],
            )

            if not sklearn_check_version("1.5") and sklearn_check_version("1.3"):
                if isinstance(sample_weight, str) and sample_weight == "deprecated":
                    sample_weight = None

                if sample_weight is not None:
                    warnings.warn(
                        "'sample_weight' was deprecated in version 1.3 and "
                        "will be removed in 1.5.",
                        FutureWarning,
                    )

            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.cluster_centers_ = self.cluster_centers_

            return self._onedal_estimator.score(X, queue=queue)

        def _save_attributes(self):
            assert hasattr(self, "_onedal_estimator")
            self.cluster_centers_ = self._onedal_estimator.cluster_centers_
            self.labels_ = self._onedal_estimator.labels_
            self.inertia_ = self._onedal_estimator.inertia_
            self.n_iter_ = self._onedal_estimator.n_iter_
            self.n_features_in_ = self._onedal_estimator.n_features_in_

            self._n_init = self._onedal_estimator._n_init

        fit.__doc__ = _sklearn_KMeans.fit.__doc__
        predict.__doc__ = _sklearn_KMeans.predict.__doc__
        score.__doc__ = _sklearn_KMeans.score.__doc__

else:
    from daal4py.sklearn.cluster import KMeans

    logging.warning(
        "Sklearnex KMeans requires oneDAL version >= 2023.2, falling back to daal4py."
    )
