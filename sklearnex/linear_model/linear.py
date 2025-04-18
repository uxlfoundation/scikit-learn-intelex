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

import logging

import numpy as np
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, get_patch_message, register_hyperparameters
from ..base import oneDALEstimator
from ..utils.validation import validate_data

if not sklearn_check_version("1.2"):
    from sklearn.linear_model._base import _deprecate_normalize

from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted, check_X_y

from onedal.common.hyperparameters import get_hyperparameters
from onedal.linear_model import LinearRegression as onedal_LinearRegression
from onedal.utils.validation import _num_features, _num_samples


@register_hyperparameters({"fit": get_hyperparameters("linear_regression", "train")})
@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class LinearRegression(oneDALEstimator, _sklearn_LinearRegression):
    __doc__ = _sklearn_LinearRegression.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_LinearRegression._parameter_constraints
        }

        if sklearn_check_version("1.7"):

            def __init__(
                self,
                fit_intercept=True,
                copy_X=True,
                tol=1e-06,  # for sparse solver only, not used by oneDAL
                n_jobs=None,
                positive=False,
            ):
                super().__init__(
                    fit_intercept=fit_intercept,
                    copy_X=copy_X,
                    tol=tol,
                    n_jobs=n_jobs,
                    positive=positive,
                )

        else:

            def __init__(
                self,
                fit_intercept=True,
                copy_X=True,
                n_jobs=None,
                positive=False,
            ):
                super().__init__(
                    fit_intercept=fit_intercept,
                    copy_X=copy_X,
                    n_jobs=n_jobs,
                    positive=positive,
                )

    else:

        def __init__(
            self,
            fit_intercept=True,
            normalize="deprecated",
            copy_X=True,
            n_jobs=None,
            positive=False,
        ):
            super().__init__(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive,
            )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        # It is necessary to properly update coefs for predict if we
        # fallback to sklearn in dispatch
        if hasattr(self, "_onedal_estimator"):
            del self._onedal_estimator

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_LinearRegression.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_LinearRegression.predict,
            },
            X,
        )

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": _sklearn_LinearRegression.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    def _onedal_cpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
        )
        return self._onedal_supported(patching_status, method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
        )

        if method_name == "fit" and not daal_check_version((2025, "P", 200)):
            n_samples = _num_samples(data[0])
            n_features = _num_features(data[0], fallback_1d=True)
            is_underdetermined = n_samples < (n_features + int(self.fit_intercept))
            patching_status.and_conditions(
                [
                    (
                        not is_underdetermined,
                        "The shape of X (fitting) does not satisfy oneDAL requirements:"
                        "Number of features + 1 >= number of samples.",
                    )
                ]
            )

        return self._onedal_supported(patching_status, method_name, *data)

    def _onedal_supported(self, patching_status, method_name, *data):
        if method_name == "fit":
            return self._onedal_fit_supported(patching_status, method_name, *data)
        if method_name in ["predict", "score"]:
            return self._onedal_predict_supported(patching_status, method_name, *data)
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_fit_supported(self, patching_status, method_name, *data):
        assert method_name == "fit"
        assert len(data) == 3
        X, y, sample_weight = data

        normalize_is_set = (
            hasattr(self, "normalize")
            and self.normalize
            and self.normalize != "deprecated"
        )
        positive_is_set = hasattr(self, "positive") and self.positive

        n_samples = _num_samples(X)
        n_features = _num_features(X, fallback_1d=True)

        # Note: support for some variants was either introduced in oneDAL 2025.1,
        # or had bugs in some uncommon cases in older versions.
        is_underdetermined = n_samples < (n_features + int(self.fit_intercept))
        supports_all_variants = daal_check_version((2025, "P", 1))
        is_multi_output = _num_features(y, fallback_1d=True) > 1

        patching_status.and_conditions(
            [
                (sample_weight is None, "Sample weight is not supported."),
                (
                    not issparse(X) and not issparse(y),
                    "Sparse input is not supported.",
                ),
                (not normalize_is_set, "Normalization is not supported."),
                (
                    not positive_is_set,
                    "Forced positive coefficients are not supported.",
                ),
                (
                    not is_underdetermined or supports_all_variants,
                    "The shape of X (fitting) does not satisfy oneDAL requirements:"
                    "Number of features + 1 >= number of samples.",
                ),
                (
                    not is_multi_output or supports_all_variants,
                    "Multi-output regression is not supported.",
                ),
            ]
        )

        return patching_status

    def _onedal_predict_supported(self, patching_status, method_name, *data):
        n_samples = _num_samples(data[0])
        model_is_sparse = issparse(self.coef_) or (
            self.fit_intercept and issparse(self.intercept_)
        )
        patching_status.and_conditions(
            [
                (n_samples > 0, "Number of samples is less than 1."),
                (not issparse(data[0]), "Sparse input is not supported."),
                (not model_is_sparse, "Sparse coefficients are not supported."),
            ]
        )

        return patching_status

    def _initialize_onedal_estimator(self):
        onedal_params = {"fit_intercept": self.fit_intercept, "copy_X": self.copy_X}
        self._onedal_estimator = onedal_LinearRegression(**onedal_params)

    def _onedal_fit(self, X, y, sample_weight, queue=None):
        assert sample_weight is None

        supports_multi_output = daal_check_version((2025, "P", 1))
        X, y = validate_data(
            self,
            X=X,
            y=y,
            dtype=[np.float64, np.float32],
            accept_sparse=["csr", "csc", "coo"],
            y_numeric=True,
            multi_output=supports_multi_output,
        )

        if not sklearn_check_version("1.2"):
            self._normalize = _deprecate_normalize(
                self.normalize,
                default=False,
                estimator_name=self.__class__.__name__,
            )

        self._initialize_onedal_estimator()
        # TODO:
        # impl wrapper/primitive for this case.
        if get_config()["allow_sklearn_after_onedal"]:
            try:
                self._onedal_estimator.fit(X, y, queue=queue)
                self._save_attributes()

            except RuntimeError:
                logging.getLogger("sklearnex").info(
                    f"{self.__class__.__name__}.fit "
                    + get_patch_message("sklearn_after_onedal")
                )

                del self._onedal_estimator
                super().fit(X, y)
        else:
            self._onedal_estimator.fit(X, y, queue=queue)
            self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        X = validate_data(self, X, accept_sparse=False, reset=False)

        if not hasattr(self, "_onedal_estimator"):
            self._initialize_onedal_estimator()
            self._onedal_estimator.coef_ = self.coef_
            self._onedal_estimator.intercept_ = self.intercept_

        res = self._onedal_estimator.predict(X, queue=queue)
        return res

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    @property
    def coef_(self):
        return self._coef_

    @coef_.setter
    def coef_(self, value):
        self._coef_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.coef_ = value
            del self._onedal_estimator._onedal_model

    @coef_.deleter
    def coef_(self):
        del self._coef_

    @property
    def intercept_(self):
        return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.intercept_ = value
            del self._onedal_estimator._onedal_model

    @intercept_.deleter
    def intercept_(self):
        del self._intercept_

    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self._sparse = False
        self._coef_ = self._onedal_estimator.coef_
        self._intercept_ = self._onedal_estimator.intercept_

    fit.__doc__ = _sklearn_LinearRegression.fit.__doc__
    predict.__doc__ = _sklearn_LinearRegression.predict.__doc__
    score.__doc__ = _sklearn_LinearRegression.score.__doc__
