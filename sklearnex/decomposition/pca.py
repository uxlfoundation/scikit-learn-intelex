# ===============================================================================
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
# ===============================================================================

import logging

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 100)):
    import numbers
    from math import sqrt
    from warnings import warn

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.utils.validation import check_array, check_is_fitted

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import sklearn_check_version

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain, register_hyperparameters
    from ..base import oneDALEstimator
    from ..utils._array_api import get_namespace
    from ..utils.validation import validate_data

    if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
        from sklearn.utils import check_scalar

    if sklearn_check_version("1.2"):
        from sklearn.utils._param_validation import StrOptions

    from sklearn.decomposition import PCA as _sklearn_PCA

    from onedal.common.hyperparameters import get_hyperparameters
    from onedal.decomposition import PCA as onedal_PCA
    from onedal.utils._array_api import _is_numpy_namespace

    @register_hyperparameters({"fit": get_hyperparameters("pca", "train")})
    @control_n_jobs(decorated_methods=["fit", "transform", "fit_transform"])
    class PCA(oneDALEstimator, _sklearn_PCA):
        __doc__ = _sklearn_PCA.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**_sklearn_PCA._parameter_constraints}
            # "onedal_svd" solver uses oneDAL's PCA-SVD algorithm
            # and required for testing purposes to fully enable it in future.
            # "covariance_eigh" solver is added for ability to explicitly request
            # oneDAL's PCA-Covariance algorithm using any sklearn version < 1.5.
            _parameter_constraints["svd_solver"] = [
                StrOptions(
                    _parameter_constraints["svd_solver"][0].options
                    | {"onedal_svd", "covariance_eigh"}
                )
            ]

        if sklearn_check_version("1.1"):

            def __init__(
                self,
                n_components=None,
                *,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                n_oversamples=10,
                power_iteration_normalizer="auto",
                random_state=None,
            ):
                self.n_components = n_components
                self.copy = copy
                self.whiten = whiten
                self.svd_solver = svd_solver
                self.tol = tol
                self.iterated_power = iterated_power
                self.n_oversamples = n_oversamples
                self.power_iteration_normalizer = power_iteration_normalizer
                self.random_state = random_state

        else:

            def __init__(
                self,
                n_components=None,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                random_state=None,
            ):
                self.n_components = n_components
                self.copy = copy
                self.whiten = whiten
                self.svd_solver = svd_solver
                self.tol = tol
                self.iterated_power = iterated_power
                self.random_state = random_state

        def fit(self, X, y=None):
            self._fit(X)
            return self

        @wrap_output_data
        def _fit(self, X):
            if sklearn_check_version("1.2"):
                self._validate_params()
            elif sklearn_check_version("1.1"):
                check_scalar(
                    self.n_oversamples,
                    "n_oversamples",
                    min_val=1,
                    target_type=numbers.Integral,
                )

            return dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": _sklearn_PCA._fit,
                },
                X,
            )

        def _onedal_fit(self, X, queue=None):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                copy=self.copy,
            )

            onedal_params = {
                "n_components": self.n_components,
                "is_deterministic": True,
                "method": "svd" if self._fit_svd_solver == "onedal_svd" else "cov",
                "whiten": self.whiten,
            }
            self._onedal_estimator = onedal_PCA(**onedal_params)
            self._onedal_estimator.fit(X, queue=queue)
            self._save_attributes()

            U = None
            S = self.singular_values_
            Vt = self.components_

            if sklearn_check_version("1.5"):
                xp, _ = get_namespace(X)
                x_is_centered = not self.copy

                return U, S, Vt, X, x_is_centered, xp
            else:
                return U, S, Vt

        @wrap_output_data
        def transform(self, X):
            check_is_fitted(self)
            return dispatch(
                self,
                "transform",
                {
                    "onedal": self.__class__._onedal_transform,
                    "sklearn": _sklearn_PCA.transform,
                },
                X,
            )

        def _onedal_transform(self, X, queue=None):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                reset=False,
            )
            self._validate_n_features_in_after_fitting(X)

            return self._onedal_estimator.predict(X, queue=queue)

        def fit_transform(self, X, y=None):
            if sklearn_check_version("1.5"):
                U, S, Vt, X_fit, x_is_centered, xp = self._fit(X)
            else:
                U, S, Vt = self._fit(X)
                X_fit = X
            if hasattr(self, "_onedal_estimator"):
                # oneDAL PCA was fit
                return self.transform(X)
            elif U is not None:
                # Scikit-learn PCA was fit
                U = U[:, : self.n_components_]

                if self.whiten:
                    U *= sqrt(X_fit.shape[0] - 1)
                else:
                    U *= S[: self.n_components_]

                return U
            else:
                # Scikit-learn PCA["covariance_eigh"] was fit
                return self._transform(X_fit, xp, x_is_centered=x_is_centered)

        def inverse_transform(self, X):
            # sklearn does not properly input check inverse_transform using
            # ``validate_data`` (as of sklearn 1.7). Yielding the namespace
            # in this way will conform to sklearn and various inputs without
            # causing issues with dimensionality checks, array api support,
            # etc. evaluated in ``validate_data``. This is a special solution.
            xp = (
                func()
                if (func := getattr(X, "__array_namespace__", None))
                else get_namespace(X)[0]
            )

            mean = self.mean_
            if self.whiten:
                components = (
                    xp.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_
                )
            else:
                components = self.components_

            if not _is_numpy_namespace(xp):
                # Force matching type to input data if possible
                components = xp.asarray(components, device=X.device)
                mean = xp.asarray(mean, device=X.device)

            return X @ components + mean

        def _onedal_supported(self, method_name, X):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.decomposition.{class_name}.{method_name}"
            )

            if method_name == "fit":
                shape_tuple, _is_shape_compatible = self._get_shape_compatibility(X)
                patching_status.and_conditions(
                    [
                        (
                            _is_shape_compatible,
                            "Data shape is not compatible.",
                        ),
                        (
                            self._is_solver_compatible_with_onedal(shape_tuple),
                            (
                                "Only 'covariance_eigh' and 'onedal_svd' "
                                "solvers are supported."
                                if sklearn_check_version("1.5")
                                else "Only 'full', 'covariance_eigh' and 'onedal_svd' "
                                "solvers are supported."
                            ),
                        ),
                        (not issparse(X), "oneDAL PCA does not support sparse data"),
                    ]
                )
                return patching_status

            if method_name == "transform":
                patching_status.and_conditions(
                    [
                        (
                            hasattr(self, "_onedal_estimator"),
                            "oneDAL model was not trained",
                        ),
                    ]
                )
                return patching_status

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _get_shape_compatibility(self, X):
            _is_shape_compatible = False
            _empty_shape = (0, 0)
            if hasattr(X, "shape"):
                shape_tuple = X.shape
                if len(shape_tuple) == 1:
                    shape_tuple = (1, shape_tuple[0])
            elif isinstance(X, list):
                if np.ndim(X) == 1:
                    shape_tuple = (1, len(X))
                elif np.ndim(X) == 2:
                    shape_tuple = (len(X), len(X[0]))
            else:
                return _empty_shape, _is_shape_compatible

            if shape_tuple[0] > 0 and shape_tuple[1] > 0 and len(shape_tuple) == 2:
                _is_shape_compatible = shape_tuple[1] / shape_tuple[0] < 2

            return shape_tuple, _is_shape_compatible

        def _is_solver_compatible_with_onedal(self, shape_tuple):
            self._fit_svd_solver = self.svd_solver
            n_sf_min = min(shape_tuple)
            n_components = n_sf_min if self.n_components is None else self.n_components

            if self._fit_svd_solver == "auto":
                if sklearn_check_version("1.1"):
                    if (
                        sklearn_check_version("1.5")
                        and shape_tuple[1] <= 1_000
                        and shape_tuple[0] >= 10 * shape_tuple[1]
                    ):
                        self._fit_svd_solver = "covariance_eigh"
                    elif max(shape_tuple) <= 500 or n_components == "mle":
                        self._fit_svd_solver = "full"
                    elif 1 <= n_components < 0.8 * n_sf_min:
                        self._fit_svd_solver = "randomized"
                    else:
                        self._fit_svd_solver = "full"
                else:
                    if n_components == "mle":
                        self._fit_svd_solver = "full"
                    else:
                        # check if sklearnex is faster than randomized sklearn
                        # Refer to daal4py
                        regression_coefs = np.array(
                            [
                                [
                                    9.779873e-11,
                                    shape_tuple[0] * shape_tuple[1] * n_components,
                                ],
                                [
                                    -1.122062e-11,
                                    shape_tuple[0] * shape_tuple[1] * shape_tuple[1],
                                ],
                                [1.127905e-09, shape_tuple[0] ** 2],
                            ]
                        )
                        if (
                            n_components >= 1
                            and np.dot(regression_coefs[:, 0], regression_coefs[:, 1])
                            <= 0
                        ):
                            self._fit_svd_solver = "randomized"
                        else:
                            self._fit_svd_solver = "full"

            # Use oneDAL in next cases:
            # 1. oneDAL SVD solver is explicitly set
            # 2. solver is set or dispatched to "covariance_eigh"
            # 3. solver is set or dispatched to "full" and sklearn version < 1.5
            # 4. solver is set to "auto" and dispatched to "full"
            if self._fit_svd_solver in ["onedal_svd", "covariance_eigh"]:
                return True
            elif not sklearn_check_version("1.5") and self._fit_svd_solver == "full":
                self._fit_svd_solver = "covariance_eigh"
                return True
            elif self.svd_solver == "auto" and self._fit_svd_solver == "full":
                warn(
                    "Sklearnex always uses `covariance_eigh` solver instead of `full` "
                    "when `svd_solver` parameter is set to `auto` "
                    "for performance purposes."
                )
                self._fit_svd_solver = "covariance_eigh"
                return True
            else:
                return False

        def _save_attributes(self):
            self.n_samples_ = self._onedal_estimator.n_samples_
            if sklearn_check_version("1.2"):
                self.n_features_in_ = self._onedal_estimator.n_features_
            else:
                self.n_features_ = self._onedal_estimator.n_features_
                self.n_features_in_ = self._onedal_estimator.n_features_
            self.n_components_ = self._onedal_estimator.n_components_
            self.components_ = self._onedal_estimator.components_
            self.mean_ = self._onedal_estimator.mean_
            self.singular_values_ = self._onedal_estimator.singular_values_
            self.explained_variance_ = self._onedal_estimator.explained_variance_.ravel()
            self.explained_variance_ratio_ = (
                self._onedal_estimator.explained_variance_ratio_
            )
            self.noise_variance_ = self._onedal_estimator.noise_variance_

        def _validate_n_features_in_after_fitting(self, X):
            if sklearn_check_version("1.2"):
                expected_n_features = self.n_features_in_
            else:
                expected_n_features = self.n_features_
            if X.shape[1] != expected_n_features:
                raise ValueError(
                    (
                        f"X has {X.shape[1]} features, "
                        f"but PCA is expecting {expected_n_features} features as input"
                    )
                )

        fit.__doc__ = _sklearn_PCA.fit.__doc__
        transform.__doc__ = _sklearn_PCA.transform.__doc__
        fit_transform.__doc__ = _sklearn_PCA.fit_transform.__doc__
        inverse_transform.__doc__ = _sklearn_PCA.inverse_transform.__doc__

else:
    from daal4py.sklearn.decomposition import PCA

    logging.warning(
        "Sklearnex PCA requires oneDAL version >= 2024.1.0 but it was not found"
    )
