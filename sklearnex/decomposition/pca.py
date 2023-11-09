#!/usr/bin/env python
# ===============================================================================
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
# ===============================================================================

import numbers
from math import sqrt

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils import _check_array

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain

if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar
if sklearn_check_version("0.23"):
    from sklearn.decomposition._pca import _infer_dimension
else:
    from sklearn.decomposition._pca import _infer_dimension_

from sklearn.decomposition import PCA as sklearn_PCA

from onedal.decomposition import PCA as onedal_PCA


class PCA(sklearn_PCA):
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**sklearn_PCA._parameter_constraints}

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

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        return dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_PCA._fit,
            },
            X,
        )
        # return self

    def transform(self, X, y=None):
        return dispatch(
            self,
            "transform",
            {
                "onedal": self.__class__._onedal_transform,
                "sklearn": sklearn_PCA.transform,
            },
            X,
        )

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def _is_onedal_compatible(self, shape_tuple):
        self._fit_svd_solver = self.svd_solver
        n_sf_min = min(shape_tuple)
        n_components = n_sf_min if self.n_components is None else self.n_components

        if self._fit_svd_solver == "auto":
            if sklearn_check_version("1.1"):
                if max(shape_tuple) <= 500 or n_components == "mle":
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
                        and np.dot(regression_coefs[:, 0], regression_coefs[:, 1]) <= 0
                    ):
                        self._fit_svd_solver = "randomized"
                    else:
                        self._fit_svd_solver = "full"

        shape_good_for_daal = shape_tuple[1] / shape_tuple[0] < 2
        if self._fit_svd_solver == "full" and shape_good_for_daal:
            return True
        else:
            return False

    def _onedal_supported(self, method_name, *data):
        X = data[0]
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.decomposition.{class_name}.{method_name}"
        )
        if method_name == "fit":
            patching_status.and_conditions(
                [
                    (
                        self._is_onedal_compatible(X.shape),
                        f"Only 'full' svd solver and data with shape "
                        "X.shape[1] < 2 * X.shape[0] are supported.",
                    ),
                ]
            )
            return patching_status

        if method_name == "transform":
            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained"),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _set_n_components_for_onedal(self, shape_tuple):
        (
            n_samples,
            n_features,
        ) = shape_tuple
        if self.n_components is None:
            n_components = min(shape_tuple)
        else:
            n_components = self.n_components

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(shape_tuple):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(shape_tuple))
            )
        # elif n_components >= 1:
        #     if not isinstance(n_components, numbers.Integral):
        #         raise ValueError(
        #             "n_components=%r must be of type int "
        #             "when greater than or equal to 1, "
        #             "was of type=%r" % (n_components, type(n_components))
        #         )

        if n_components == "mle":
            onedal_n_components = min(shape_tuple)
        elif 0 < n_components < 1:
            onedal_n_components = min(shape_tuple)
        else:
            onedal_n_components = n_components
        return onedal_n_components

    def _save_attributes(self):
        self.n_samples_ = self._onedal_estimator.n_samples_

        if sklearn_check_version("1.2"):
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            n_features = self.n_features_in_
        elif sklearn_check_version("0.24"):
            self.n_features_ = self._onedal_estimator.n_features_
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            n_features = self.n_features_in_
        else:
            self.n_features_ = self._onedal_estimator.n_features_
            n_features = self.n_features_
        n_sf_min = min(self.n_samples_, n_features)

        self.mean_ = self._onedal_estimator.mean_
        self.singular_values_ = self._onedal_estimator.singular_values_
        self.explained_variance_ = self._onedal_estimator.explained_variance_
        self.explained_variance_ratio_ = self._onedal_estimator.explained_variance_ratio_

        if self.n_components is None:
            self.n_components_ = self._onedal_estimator.n_components_
        elif self.n_components == "mle":
            if sklearn_check_version("0.23"):
                self.n_components_ = _infer_dimension(
                    self.explained_variance_, self.n_samples_
                )
            else:
                self.n_components_ = _infer_dimension_(
                    self.explained_variance_, self.n_samples_, n_features
                )
        elif 0 < self.n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            self.n_components_ = (
                np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
            )
        else:
            self.n_components_ = self._onedal_estimator.n_components_

        if self.n_components_ < n_sf_min:
            if self.explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = self.explained_variance_[
                    self.n_components_ :
                ].mean()
            else:
                self.noise_variance_ = self._onedal_estimator.noise_variance_
        else:
            self.noise_variance_ = 0.0

        self.explained_variance_ = self.explained_variance_[: self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            : self.n_components_
        ]
        self.components_ = self._onedal_estimator.components_[: self.n_components_]
        self.singular_values_ = self.singular_values_[: self.n_components_]

    def _onedal_fit(self, X, queue=None):
        if issparse(X):
            raise TypeError(
                "PCA does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )
        # if sklearn_check_version("1.2"):
        #     _parameter_constraints: dict = {**sklearn_PCA._parameter_constraints}
        # # elif sklearn_check_version("1.1"):
        # #     check_scalar(
        # #         self.n_oversamples,
        # #         "n_oversamples",
        # #         min_val=1,
        # #         target_type=numbers.Integral,
        # #     )

        if sklearn_check_version("0.23"):
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )
        else:
            X = _check_array(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )

        onedal_n_components = self._set_n_components_for_onedal(X.shape)

        onedal_params = {
            "n_components": onedal_n_components,
            "is_deterministic": True,
            "method": "precomputed",
            "whiten": self.whiten,
        }
        self._onedal_estimator = onedal_PCA(**onedal_params)
        self._onedal_estimator.fit(X, queue=queue)
        self._save_attributes()
        return self

    def _validate_n_features_in(self, X):
        if hasattr(self, "n_features_in_"):
            if self.n_features_in_ != X.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features, "
                    f"but {self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features as input"
                )
        elif hasattr(self, "n_features_"):
            if self.n_features_ != X.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features, "
                    f"but {self.__class__.__name__} is expecting "
                    f"{self.n_features_} features as input"
                )

    def _onedal_transform(self, X, queue=None):
        check_is_fitted(self)
        self._validate_n_features_in(X)
        if sklearn_check_version("0.23"):
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )
        else:
            X = _check_array(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )
        return self._onedal_estimator.predict(X, queue=queue)

    fit.__doc__ = sklearn_PCA.fit.__doc__
    transform.__doc__ = sklearn_PCA.transform.__doc__
    fit_transform.__doc__ = sklearn_PCA.fit_transform.__doc__
