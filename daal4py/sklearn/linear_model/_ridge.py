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

import logging
import numbers

import numpy as np
from scipy import sparse as sp
from sklearn.linear_model._ridge import Ridge as Ridge_original
from sklearn.linear_model._ridge import _BaseRidge
from sklearn.utils import check_array, check_X_y

import daal4py

from .._n_jobs_support import control_n_jobs
from .._utils import (
    PatchingConditionsChain,
    get_patch_message,
    getFPType,
    make2d,
    sklearn_check_version,
)
from ..utils.validation import validate_data

if not sklearn_check_version("1.2"):
    from sklearn.linear_model._base import _deprecate_normalize
if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar


def _daal4py_fit(self, X, y_):
    X = make2d(X)
    y = make2d(y_)

    _fptype = getFPType(X)

    ridge_params = np.asarray(self.alpha, dtype=X.dtype)
    if ridge_params.size != 1 and ridge_params.size != y.shape[1]:
        # incorrect order of parameters in the error message is intentional to match sklearn
        raise ValueError(
            "Number of targets and number of penalties do not correspond: "
            f"{ridge_params.size} != {y.shape[1]}"
        )
    ridge_params = ridge_params.reshape((-1, 1))

    ridge_alg = daal4py.ridge_regression_training(
        fptype=_fptype,
        method="defaultDense",
        interceptFlag=(self.fit_intercept is True),
        ridgeParameters=ridge_params,
    )

    try:
        ridge_res = ridge_alg.compute(X, y)
    except RuntimeError:
        return None

    ridge_model = ridge_res.model
    self.daal_model_ = ridge_model
    coefs = ridge_model.Beta

    self.intercept_ = coefs[:, 0].copy(order="C")
    self.coef_ = coefs[:, 1:].copy(order="C")

    if self.coef_.shape[0] == 1 and y_.ndim == 1:
        self.coef_ = np.ravel(self.coef_)
        self.intercept_ = self.intercept_[0]

    return self


def _daal4py_predict(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)

    ridge_palg = daal4py.ridge_regression_prediction(
        fptype=_fptype, method="defaultDense"
    )
    if self.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, "
            f"but Ridge is expecting {self.n_features_in_} features as input"
        )
    ridge_res = ridge_palg.compute(X, self.daal_model_)

    res = ridge_res.prediction

    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)
    return res


def _fit_ridge(self, _X, _y, sample_weight=None):
    if not sklearn_check_version("1.2"):
        self._normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )
    if sklearn_check_version("1.2"):
        self._validate_params()
    elif sklearn_check_version("1.1"):
        if self.max_iter is not None:
            self.max_iter = check_scalar(
                self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
            )
        self.tol = check_scalar(self.tol, "tol", target_type=numbers.Real, min_val=0.0)
        if self.alpha is not None and not isinstance(self.alpha, (np.ndarray, tuple)):
            self.alpha = check_scalar(
                self.alpha,
                "alpha",
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries="left",
            )

    X, y = validate_data(
        self,
        _X,
        _y,
        accept_sparse=["csr", "csc", "coo"],
        dtype=[np.float64, np.float32],
        multi_output=True,
        y_numeric=True,
        ensure_2d=True,
    )

    self.sample_weight_ = sample_weight
    self.fit_shape_good_for_daal_ = True if X.shape[0] >= X.shape[1] else False

    _patching_status = PatchingConditionsChain("sklearn.linear_model.Ridge.fit")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                self.solver == "auto",
                f"'{self.solver}' solver is not supported. "
                "Only 'auto' solver is supported.",
            ),
            (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                self.fit_shape_good_for_daal_,
                "The shape of X does not satisfy oneDAL requirements: "
                "number of features > number of samples.",
            ),
            (
                X.dtype == np.float64 or X.dtype == np.float32,
                f"'{X.dtype}' X data type is not supported. "
                "Only np.float32 and np.float64 are supported.",
            ),
            (sample_weight is None, "Sample weights are not supported."),
            (
                not (hasattr(self, "positive") and self.positive),
                "Forced positive coefficients are not supported.",
            ),
        ]
    )
    _patching_status.write_log()

    if not _dal_ready:
        if hasattr(self, "daal_model_"):
            del self.daal_model_
        return Ridge_original.fit(self, _X, _y, sample_weight=sample_weight)
    self.n_iter_ = None
    res = _daal4py_fit(self, X, y)
    if res is None:
        logging.info(
            "sklearn.linear_model.Ridge.fit: " + get_patch_message("sklearn_after_daal")
        )
        if hasattr(self, "daal_model_"):
            del self.daal_model_
        return Ridge_original.fit(self, _X, _y, sample_weight=sample_weight)
    return res


def _predict_ridge(self, _X):
    X = validate_data(
        self,
        _X,
        accept_sparse=["csr", "csc", "coo"],
        dtype=[np.float64, np.float32],
        reset=False,
        ensure_2d=True,
    )
    good_shape_for_daal = (
        True if X.ndim <= 1 else True if X.shape[0] >= X.shape[1] else False
    )

    _patching_status = PatchingConditionsChain("sklearn.linear_model.Ridge.predict")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                self.solver == "auto",
                f"'{self.solver}' solver is not supported. "
                "Only 'auto' solver is supported.",
            ),
            (hasattr(self, "daal_model_"), "oneDAL model was not trained."),
            (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                good_shape_for_daal,
                "The shape of X does not satisfy oneDAL requirements: "
                "number of features > number of samples.",
            ),
            (
                X.dtype == np.float64 or X.dtype == np.float32,
                f"'{X.dtype}' X data type is not supported. "
                "Only np.float32 and np.float64 are supported.",
            ),
            (
                not hasattr(self, "sample_weight_") or self.sample_weight_ is None,
                "Sample weights are not supported.",
            ),
        ]
    )
    _patching_status.write_log()

    if not _dal_ready:
        return self._decision_function(_X)
    return _daal4py_predict(self, X)


@control_n_jobs(decorated_methods=["fit", "predict"])
class Ridge(Ridge_original, _BaseRidge):
    __doc__ = Ridge_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**Ridge_original._parameter_constraints}

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
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.copy_X = copy_X
            self.max_iter = max_iter
            self.tol = tol
            self.solver = solver
            self.positive = positive
            self.random_state = random_state

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
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.normalize = normalize
            self.copy_X = copy_X
            self.max_iter = max_iter
            self.tol = tol
            self.solver = solver
            self.positive = positive
            self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        return _fit_ridge(self, X, y, sample_weight=sample_weight)

    def predict(self, X):
        return _predict_ridge(self, X)

    fit.__doc__ = Ridge_original.fit.__doc__
    predict.__doc__ = Ridge_original.predict.__doc__
