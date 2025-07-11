# ===============================================================================
# Copyright 2024 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

if daal_check_version((2024, "P", 600)):
    import numbers

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge as _sklearn_Ridge
    from sklearn.metrics import r2_score
    from sklearn.utils.validation import check_is_fitted

    from daal4py.sklearn._n_jobs_support import control_n_jobs

    if not sklearn_check_version("1.2"):
        from sklearn.linear_model._base import _deprecate_normalize
    if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
        from sklearn.utils import check_scalar

    from onedal.linear_model import Ridge as onedal_Ridge
    from onedal.utils.validation import _num_features, _num_samples

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain
    from ..base import oneDALEstimator
    from ..utils.validation import validate_data

    @control_n_jobs(decorated_methods=["fit", "predict", "score"])
    class Ridge(oneDALEstimator, _sklearn_Ridge):
        __doc__ = _sklearn_Ridge.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**_sklearn_Ridge._parameter_constraints}

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                copy_X=True,
                max_iter=None,
                tol=1e-4,
                solver="auto",
                positive=False,
                random_state=None,
            ):
                super().__init__(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    copy_X=copy_X,
                    max_iter=max_iter,
                    tol=tol,
                    solver=solver,
                    positive=positive,
                    random_state=random_state,
                )

        else:

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                max_iter=None,
                tol=1e-3,
                solver="auto",
                positive=False,
                random_state=None,
            ):
                super().__init__(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                    copy_X=copy_X,
                    max_iter=max_iter,
                    solver=solver,
                    tol=tol,
                    positive=positive,
                    random_state=random_state,
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
                    "sklearn": _sklearn_Ridge.fit,
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
                    "sklearn": _sklearn_Ridge.predict,
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
                    "sklearn": _sklearn_Ridge.score,
                },
                X,
                y,
                sample_weight=sample_weight,
            )

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

            patching_status.and_conditions(
                [
                    (
                        self.solver == "auto",
                        f"'{self.solver}' solver is not supported. "
                        "Only 'auto' solver is supported.",
                    ),
                    (
                        not issparse(X) and not issparse(y),
                        "Sparse input is not supported.",
                    ),
                    (sample_weight is None, "Sample weight is not supported."),
                    (not normalize_is_set, "Normalization is not supported."),
                    (
                        not positive_is_set,
                        "Forced positive coefficients are not supported.",
                    ),
                    (
                        isinstance(self.alpha, numbers.Real),
                        "Non-scalar alpha is not supported yet.",
                    ),
                ]
            )

            return patching_status

        def _onedal_predict_supported(self, patching_status, method_name, *data):
            assert method_name in ["predict", "score"]
            assert len(data) <= 2

            n_samples = _num_samples(data[0])
            model_is_sparse = issparse(self.coef_) or (
                self.fit_intercept and issparse(self.intercept_)
            )
            patching_status.and_conditions(
                [
                    (
                        self.solver == "auto",
                        f"'{self.solver}' solver is not supported. "
                        "Only 'auto' solver is supported.",
                    ),
                    (n_samples > 0, "Number of samples is less than 1."),
                    (not issparse(data[0]), "Sparse input is not supported."),
                    (not model_is_sparse, "Sparse coefficients are not supported."),
                ]
            )

            return patching_status

        def _onedal_gpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
            )

            if method_name == "fit":
                if not daal_check_version((2025, "P", 200)):
                    n_samples = _num_samples(data[0])
                    n_features = _num_features(data[0], fallback_1d=True)
                    is_underdetermined = n_samples < (
                        n_features + int(self.fit_intercept)
                    )
                    is_zero_alpha = isinstance(self.alpha, numbers.Real) and np.isclose(
                        self.alpha, 0, atol=1e-5
                    )

                    patching_status.and_condition(
                        not is_underdetermined or not is_zero_alpha,
                        "The shape of X (fitting) does not satisfy oneDAL requirements:"
                        "Number of features + 1 >= number of samples and alpha = 0.",
                    )

                return self._onedal_fit_supported(patching_status, method_name, *data)

            if method_name in ["predict", "score"]:
                return self._onedal_predict_supported(patching_status, method_name, *data)

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_cpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
            )

            if method_name == "fit":
                if not daal_check_version((2025, "P", 100)):
                    n_samples = _num_samples(data[0])
                    n_features = _num_features(data[0], fallback_1d=True)
                    is_underdetermined = n_samples < (
                        n_features + int(self.fit_intercept)
                    )
                    is_zero_alpha = isinstance(self.alpha, numbers.Real) and np.isclose(
                        self.alpha, 0, atol=1e-5
                    )

                    patching_status.and_condition(
                        not is_underdetermined or not is_zero_alpha,
                        "The shape of X (fitting) does not satisfy oneDAL requirements:"
                        "Number of features + 1 >= number of samples and alpha = 0.",
                    )
                return self._onedal_fit_supported(patching_status, method_name, *data)

            if method_name in ["predict", "score"]:
                return self._onedal_predict_supported(patching_status, method_name, *data)

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _initialize_onedal_estimator(self):
            onedal_params = {
                "fit_intercept": self.fit_intercept,
                "alpha": self.alpha,
                "copy_X": self.copy_X,
            }
            self._onedal_estimator = onedal_Ridge(**onedal_params)

        def _onedal_fit(self, X, y, sample_weight, queue=None):
            # `Sample weight` is not supported. Expected to be None value.
            assert sample_weight is None

            if sklearn_check_version("1.2"):
                self._validate_params()
            elif sklearn_check_version("1.1"):
                if self.max_iter is not None:
                    self.max_iter = check_scalar(
                        self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
                    )
                self.tol = check_scalar(
                    self.tol, "tol", target_type=numbers.Real, min_val=0.0
                )
                if self.alpha is not None and not isinstance(
                    self.alpha, (np.ndarray, tuple)
                ):
                    self.alpha = check_scalar(
                        self.alpha,
                        "alpha",
                        target_type=numbers.Real,
                        min_val=0.0,
                        include_boundaries="left",
                    )

            X, y = validate_data(
                self,
                X=X,
                y=y,
                dtype=[np.float64, np.float32],
                accept_sparse=["csr", "csc", "coo"],
                y_numeric=True,
                multi_output=True,
            )

            if not sklearn_check_version("1.2"):
                self._normalize = _deprecate_normalize(
                    self.normalize,
                    default=False,
                    estimator_name=self.__class__.__name__,
                )

            self._initialize_onedal_estimator()
            self._onedal_estimator.fit(X, y, queue=queue)
            self._save_attributes()

            if sklearn_check_version("1.6"):
                if y.ndim == 1 or y.shape[1] == 1:
                    self.coef_ = self.coef_.ravel()
                    self.intercept_ = self.intercept_[0]
            else:
                if self.coef_.shape[0] == 1 and y.ndim == 1:
                    self.coef_ = self.coef_.ravel()
                    self.intercept_ = self.intercept_[0]

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
            return self._coef

        @coef_.setter
        def coef_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.coef_ = value
                # checking if the model is already fitted and if so, deleting the model
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._coef = value

        @property
        def intercept_(self):
            return self._intercept

        @intercept_.setter
        def intercept_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.intercept_ = value
                # checking if the model is already fitted and if so, deleting the model
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._intercept = value

        def _save_attributes(self):
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            self._coef = self._onedal_estimator.coef_
            self._intercept = self._onedal_estimator.intercept_

        fit.__doc__ = _sklearn_Ridge.fit.__doc__
        predict.__doc__ = _sklearn_Ridge.predict.__doc__
        score.__doc__ = _sklearn_Ridge.score.__doc__

else:
    from daal4py.sklearn.linear_model import Ridge
    from onedal._device_offload import support_input_format

    Ridge.fit = support_input_format(Ridge.fit)
    Ridge.predict = support_input_format(Ridge.predict)
    Ridge.score = support_input_format(Ridge.score)

    logging.warning("Ridge requires oneDAL version >= 2024.6 but it was not found")
