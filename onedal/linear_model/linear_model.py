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
from numbers import Number

import numpy as np

from daal4py.sklearn._utils import daal_check_version, get_dtype, make2d
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from .._config import _get_config
from ..common._estimator_checks import _check_is_fitted
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import _check_array, _check_n_features, _check_X_y, _num_features


class BaseLinearRegression(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, fit_intercept, copy_X, algorithm, alpha=0.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.copy_X = copy_X
        self.algorithm = algorithm

    @bind_default_backend("linear_model.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("linear_model.regression")
    def infer(self, params, model, X): ...

    # direct access to the backend model class
    @bind_default_backend("linear_model.regression")
    def model(self): ...

    def _get_onedal_params(self, dtype=np.float32):
        intercept = "intercept|" if self.fit_intercept else ""
        params = {
            "fptype": dtype,
            "method": self.algorithm,
            "intercept": self.fit_intercept,
            "result_option": (intercept + "coefficients"),
        }
        if daal_check_version((2024, "P", 600)):
            params["alpha"] = self.alpha

        return params

    def _create_model(self):
        model = self.model()

        coefficients = self.coef_
        dtype = get_dtype(coefficients)

        if coefficients.ndim == 2:
            n_features_in = coefficients.shape[1]
            n_targets_in = coefficients.shape[0]
        else:
            n_features_in = coefficients.size
            n_targets_in = 1

        intercept = self.intercept_
        if isinstance(intercept, Number):
            assert n_targets_in == 1
        else:
            if not isinstance(intercept, np.ndarray):
                intercept = np.asarray(intercept, dtype=dtype)
            assert n_targets_in == intercept.size

        coefficients, intercept = make2d(coefficients), make2d(intercept)
        coefficients = coefficients.T if n_targets_in == 1 else coefficients

        assert coefficients.shape == (
            n_targets_in,
            n_features_in,
        ), f"{coefficients.shape}, {n_targets_in}, {n_features_in}"
        assert intercept.shape == (n_targets_in, 1), f"{intercept.shape}, {n_targets_in}"

        desired_shape = (n_targets_in, n_features_in + 1)
        packed_coefficients = np.zeros(desired_shape, dtype=dtype)

        packed_coefficients[:, 1:] = coefficients
        if self.fit_intercept:
            packed_coefficients[:, 0][:, np.newaxis] = intercept

        model.packed_coefficients = to_table(
            packed_coefficients, queue=QM.get_global_queue()
        )

        self._onedal_model = model

        return model

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

        sua_iface, xp, _ = _get_sycl_namespace(X)
        use_raw_input = _get_config().get("use_raw_input", False) is True
        if use_raw_input and sua_iface is not None:
            queue = X.sycl_queue

        if not use_raw_input:
            X = _check_array(
                X, dtype=[np.float64, np.float32], force_all_finite=False, ensure_2d=False
            )
            X = make2d(X)

        _check_n_features(self, X, False)

        if hasattr(self, "_onedal_model"):
            model = self._onedal_model
        else:
            model = self._create_model()

        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table.dtype)
        result = self.infer(params, model, X_table)
        y = from_table(result.responses, sua_iface=sua_iface, sycl_queue=queue, xp=xp)

        if y.shape[1] == 1 and self.coef_.ndim == 1:
            return xp.reshape(y, (-1,))
        else:
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

        sua_iface, xp, _ = _get_sycl_namespace(X)
        use_raw_input = _get_config().get("use_raw_input", False) is True
        if use_raw_input and sua_iface is not None:
            queue = X.sycl_queue
        if _get_config()["use_raw_input"] is False:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)

            dtype = get_dtype(X)
            if dtype not in [np.float32, np.float64]:
                dtype = np.float64
                X = X.astype(dtype, copy=self.copy_X)

            y = np.asarray(y).astype(dtype=dtype)

            X, y = _check_X_y(X, y, force_all_finite=False, accept_2d_y=True)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X_table.dtype)

        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            result = self.train(params, hparams.backend, X_table, y_table)
        else:
            result = self.train(params, X_table, y_table)

        self._onedal_model = result.model

        packed_coefficients = from_table(
            result.model.packed_coefficients, sycl_queue=queue
        )
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:],
            packed_coefficients[:, 0],
        )

        if self.coef_.shape[0] == 1 and y.ndim == 1:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_[0]

        return self


class Ridge(BaseLinearRegression):
    """Ridge Regression oneDAL implementation.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.

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
        alpha=1.0,
        fit_intercept=True,
        copy_X=False,
        *,
        algorithm="norm_eq",
    ):
        super().__init__(
            fit_intercept=fit_intercept, alpha=alpha, copy_X=copy_X, algorithm=algorithm
        )

    @supports_queue
    def fit(self, X, y, queue=None):
        """Fit Ridge regressor.

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
        X = _check_array(
            X,
            dtype=[np.float64, np.float32],
            force_all_finite=False,
            ensure_2d=False,
            copy=self.copy_X,
        )
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        dtype = get_dtype(X)
        if dtype not in [np.float32, np.float64]:
            dtype = np.float64
            X = X.astype(dtype, copy=self.copy_X)

        y = np.asarray(y).astype(dtype=dtype)

        X, y = _check_X_y(X, y, force_all_finite=False, accept_2d_y=True)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X.dtype)

        result = self.train(params, X_table, y_table)
        self._onedal_model = result.model

        packed_coefficients = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:],
            packed_coefficients[:, 0],
        )

        return self
