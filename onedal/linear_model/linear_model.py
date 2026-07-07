# ==============================================================================
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
# ==============================================================================

from abc import ABCMeta, abstractmethod

import numpy as np

from onedal import onedal_check_version

from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..common._estimator_checks import _check_is_fitted
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils.validation import _check_n_features, _num_features


class BaseLinearRegression(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, fit_intercept, copy_X, algorithm, alpha=0.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.copy_X = copy_X
        self.algorithm = algorithm
        self._onedal_model = None

    @bind_default_backend("linear_model.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("linear_model.regression")
    def infer(self, params, model, X): ...

    # direct access to the backend model class
    @bind_default_backend("linear_model.regression")
    def model(self): ...

    def _get_onedal_params(self, dtype):
        intercept = "intercept|" if self.fit_intercept else ""
        params = {
            "fptype": dtype,
            "method": self.algorithm,
            "intercept": self.fit_intercept,
            "result_option": (intercept + "coefficients"),
        }
        if onedal_check_version(2024, 6, 0):
            params["alpha"] = self.alpha

        return params

    def _create_model(self, coef_, intercept_, xp) -> None:
        model = self.model()

        dim0 = coef_.shape[0] if len(coef_.shape) > 1 else 1
        if dim0 == 1:
            coef_ = xp.reshape(coef_, (1, -1))
        if self.fit_intercept:
            intercept_ = xp.reshape(intercept_, (dim0, -1))
            if xp is np:
                # workaround for numpy<2
                packed_coefficients = xp.concatenate([intercept_, coef_], axis=1)
            else:
                packed_coefficients = xp.concat([intercept_, coef_], axis=1)
        else:
            if hasattr(coef_, "device"):
                packed_coefficients = xp.zeros(
                    (dim0, coef_.shape[1] + 1), device=coef_.device
                )
            else:
                # Note: on numpy<2, these objects and functions do not have
                # the 'device' attribute/argument.
                packed_coefficients = xp.zeros((dim0, coef_.shape[1] + 1))
            packed_coefficients[:, 1:] = coef_

        # packed_coefficients = xp.reshape(packed_coefficients, (1, -1))
        model.packed_coefficients = to_table(packed_coefficients)
        self.coef_ = coef_
        self.intercept_ = intercept_
        self._onedal_model = model

    @supports_queue
    def fit(self, X, y, queue=None):
        """Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X_table.dtype)

        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            result = self.train(params, hparams.backend, X_table, y_table)
        else:
            result = self.train(params, X_table, y_table)

        self._onedal_model = result.model

        packed_coefficients = from_table(result.model.packed_coefficients, like=X)
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:],
            packed_coefficients[:, 0],
        )

        return self

    @supports_queue
    def predict(self, X, queue=None):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        y : array, shape (n_samples, n_targets)
            Returns predicted values.
        """

        _check_is_fitted(self)

        _check_n_features(self, X, False)

        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table.dtype)
        result = self.infer(params, self._onedal_model, X_table)
        y = from_table(result.responses, like=X)

        return y


class LinearRegression(BaseLinearRegression):
    """Linear Regression oneDAL implementation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    algorithm : str, default="norm_eq"
        Algorithm used for oneDAL computation.
    """

    def __init__(
        self,
        fit_intercept=True,
        copy_X=False,
        *,
        algorithm="norm_eq",
    ):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, algorithm=algorithm)


class Ridge(BaseLinearRegression):
    """Ridge Regression oneDAL implementation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    algorithm : str, default="norm_eq"
        Algorithm used for oneDAL computation.

    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
    """

    def __init__(
        self,
        fit_intercept=True,
        copy_X=False,
        *,
        algorithm="norm_eq",
        alpha=1.0,
    ):
        super().__init__(
            fit_intercept=fit_intercept, copy_X=copy_X, algorithm=algorithm, alpha=alpha
        )
