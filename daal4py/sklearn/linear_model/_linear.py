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

import numpy as np
from scipy import sparse as sp
from sklearn.linear_model import LinearRegression as LinearRegression_original
from sklearn.utils import check_array

from .._utils import sklearn_check_version
from ..utils.validation import _daal_check_array, check_feature_names, validate_data

if not sklearn_check_version("1.2"):
    from sklearn.linear_model._base import _deprecate_normalize

import logging

import daal4py

from .._n_jobs_support import control_n_jobs
from .._utils import (
    PatchingConditionsChain,
    get_patch_message,
    getFPType,
    is_DataFrame,
    make2d,
)


def _daal4py_fit(self, X, y_):
    y = make2d(y_)
    X_fptype = getFPType(X)

    try:
        lr_algorithm = daal4py.linear_regression_training(
            fptype=X_fptype, interceptFlag=bool(self.fit_intercept), method="defaultDense"
        )
        lr_res = lr_algorithm.compute(X, y)
    except RuntimeError:
        # Normal system is not invertible, try QR
        try:
            lr_algorithm = daal4py.linear_regression_training(
                fptype=X_fptype, interceptFlag=bool(self.fit_intercept), method="qrDense"
            )
            lr_res = lr_algorithm.compute(X, y)
        except RuntimeError:
            # fall back on sklearn
            return None

    lr_model = lr_res.model
    self.daal_model_ = lr_model
    coefs = lr_model.Beta

    self.intercept_ = coefs[:, 0].copy(order="C")
    self.coef_ = coefs[:, 1:].copy(order="C")
    self.n_features_in_ = X.shape[1]
    self.rank_ = X.shape[1]
    self.singular_ = np.full((X.shape[1],), np.nan)

    if self.coef_.shape[0] == 1 and y_.ndim == 1:
        self.coef_ = np.ravel(self.coef_)
        self.intercept_ = self.intercept_[0]

    return self


def _daal4py_predict(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)
    lr_pred = daal4py.linear_regression_prediction(fptype=_fptype, method="defaultDense")
    if X.shape[1] != self.n_features_in_:
        raise ValueError(
            f"X has {X.shape[1]} features, "
            f"but LinearRegression is expecting "
            f"{self.n_features_in_} features as input"
        )
    try:
        lr_res = lr_pred.compute(X, self.daal_model_)
    except RuntimeError:
        raise ValueError(
            f"Input data shape {X.shape} is inconsistent with the trained model"
        )
    res = lr_res.prediction
    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)

    return res


def _fit_linear(self, X, y, sample_weight=None):
    params = {
        "X": X,
        "y": y,
        "accept_sparse": ["csr", "csc", "coo"],
        "y_numeric": True,
        "multi_output": True,
    }
    X, y = validate_data(
        self,
        dtype=[np.float64, np.float32],
        **params,
    )

    self.fit_shape_good_for_daal_ = bool(
        X.shape[0] > X.shape[1] + int(self.fit_intercept)
    )

    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LinearRegression.fit"
    )
    _patching_status.and_conditions(
        [
            (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                self.fit_shape_good_for_daal_,
                "The shape of X does not satisfy oneDAL requirements: "
                "number of features + 1 >= number of samples.",
            ),
            (sample_weight is None, "Sample weights are not supported."),
        ]
    )

    _dal_ready = _patching_status.get_status()
    _patching_status.write_log()
    if _dal_ready:
        res = _daal4py_fit(self, X, y)
        if res is not None:
            return res
        logging.info(
            "sklearn.linar_model.LinearRegression."
            "fit: " + get_patch_message("sklearn_after_daal")
        )

    return super(LinearRegression, self).fit(
        X,
        y,
        sample_weight=sample_weight,
    )


def _predict_linear(self, X):
    check_feature_names(self, X, reset=False)
    is_df = is_DataFrame(X)
    X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    X = np.asarray(X) if not sp.issparse(X) and not is_df else X
    good_shape_for_daal = (
        True if X.ndim <= 1 else True if X.shape[0] > X.shape[1] else False
    )

    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LinearRegression.predict"
    )
    _dal_ready = _patching_status.and_conditions(
        [
            (hasattr(self, "daal_model_"), "oneDAL model was not trained."),
            (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                good_shape_for_daal,
                "The shape of X does not satisfy oneDAL requirements: "
                "Number of features >= number of samples.",
            ),
            (
                not hasattr(self, "sample_weight_") or self.sample_weight_ is None,
                "Sample weights are not supported.",
            ),
        ]
    )
    if hasattr(self, "fit_shape_good_for_daal_"):
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    self.fit_shape_good_for_daal_,
                    "The shape of X (fitting) does not satisfy oneDAL requirements: "
                    "Number of features + 1 >= number of samples.",
                )
            ]
        )
    _patching_status.write_log()
    if not _dal_ready:
        return self._decision_function(X)
    X = _daal_check_array(X)
    return _daal4py_predict(self, X)


@control_n_jobs(decorated_methods=["fit", "predict"])
class LinearRegression(LinearRegression_original):
    __doc__ = LinearRegression_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **LinearRegression_original._parameter_constraints
        }

        def __init__(
            self,
            fit_intercept=True,
            copy_X=True,
            n_jobs=None,
            positive=False,
        ):
            super(LinearRegression, self).__init__(
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
            super(LinearRegression, self).__init__(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive,
            )

    def fit(self, X, y, sample_weight=None):
        if not sklearn_check_version("1.2"):
            self._normalize = _deprecate_normalize(
                self.normalize,
                default=False,
                estimator_name=self.__class__.__name__,
            )
        check_feature_names(self, X, reset=True)
        if sklearn_check_version("1.2"):
            self._validate_params()

        _patching_status = PatchingConditionsChain(
            "sklearn.linear_model.LinearRegression.fit"
        )
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    self.positive is False,
                    "Forced positive coefficients are not supported.",
                )
            ]
        )
        if not _dal_ready:
            _patching_status.write_log()
            return super(LinearRegression, self).fit(X, y=y, sample_weight=sample_weight)
        return _fit_linear(self, X, y, sample_weight=sample_weight)

    def predict(self, X):
        return _predict_linear(self, X)

    fit.__doc__ = LinearRegression_original.fit.__doc__
    predict.__doc__ = LinearRegression_original.predict.__doc__
