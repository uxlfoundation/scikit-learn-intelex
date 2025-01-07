# ==============================================================================
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
# ==============================================================================

import numpy as np

from daal4py.sklearn._utils import get_dtype
from onedal._device_offload import SyclQueueManager, supports_queue
from onedal.common._backend import bind_default_backend

from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils.validation import _check_X_y, _num_features
from .linear_model import BaseLinearRegression


class IncrementalLinearRegression(BaseLinearRegression):
    """
    Incremental Linear Regression oneDAL implementation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    algorithm : string, default="norm_eq"
        Algorithm used for computation on oneDAL side
    """

    def __init__(self, fit_intercept=True, copy_X=False, algorithm="norm_eq"):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, algorithm=algorithm)
        self._queue = None
        self._reset()

    @bind_default_backend("linear_model.regression")
    def partial_train_result(self): ...

    @bind_default_backend("linear_model.regression")
    def partial_train(self, *args, **kwargs): ...

    @bind_default_backend("linear_model.regression")
    def finalize_train(self, *args, **kwargs): ...

    def _reset(self):
        # Get the pointer to partial_result from backend
        self._queue = None
        self._partial_result = self.partial_train_result()

    @supports_queue
    def partial_fit(self, X, y, queue=None):
        """
        Computes partial data for linear regression
        from data batch X and saves it to `_partial_result`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y: array-like of shape (n_samples,) or (n_samples, n_targets) in
            case of multiple targets
            Responses for training data.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._queue = queue
        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32], accept_2d_y=True, force_all_finite=False
        )
        y = np.asarray(y, dtype=X.dtype)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(self._dtype)

        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            self._partial_result = self.partial_train(
                self._params, hparams.backend, self._partial_result, X_table, y_table
            )
        else:
            self._partial_result = self.partial_train(
                self._params, self._partial_result, X_table, y_table
            )
        self._queue = queue

    def finalize_fit(self):
        """
        Finalizes linear regression computation and obtains coefficients
        from the current `_partial_result`.

        Parameters
        ----------
        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        hparams = get_hyperparameters("linear_regression", "train")
        with SyclQueueManager.manage_global_queue(self._queue):
            if hparams is not None and not hparams.is_default:
                result = self.finalize_train(
                    self._params, hparams.backend, self._partial_result
                )
            else:
                result = self.finalize_train(self._params, self._partial_result)

        self._onedal_model = result.model

        packed_coefficients = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:].squeeze(),
            packed_coefficients[:, 0].squeeze(),
        )

        self._queue = None

        return self


class IncrementalRidge(BaseLinearRegression):
    """
    Incremental Ridge Regression oneDAL implementation.

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

    algorithm : string, default="norm_eq"
        Algorithm used for computation on oneDAL side
    """

    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=False, algorithm="norm_eq"):
        super().__init__(
            fit_intercept=fit_intercept, alpha=alpha, copy_X=copy_X, algorithm=algorithm
        )
        self._queue = None
        self._reset()

    def _reset(self):
        self._queue = None
        self._partial_result = self.partial_train_result()

    @bind_default_backend("linear_model.regression")
    def partial_train_result(self): ...

    @bind_default_backend("linear_model.regression")
    def partial_train(self, *args, **kwargs): ...

    @bind_default_backend("linear_model.regression")
    def finalize_train(self, *args, **kwargs): ...

    @supports_queue
    def partial_fit(self, X, y, queue=None):
        """
        Computes partial data for ridge regression
        from data batch X and saves it to `_partial_result`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y: array-like of shape (n_samples,) or (n_samples, n_targets) in
            case of multiple targets
            Responses for training data.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._queue = queue
        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32], accept_2d_y=True, force_all_finite=False
        )
        y = np.asarray(y, dtype=X.dtype)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(self._dtype)

        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            self._partial_result = self.partial_train(
                self._params, hparams.backend, self._partial_result, X_table, y_table
            )
        else:
            self._partial_result = self.partial_train(
                self._params, self._partial_result, X_table, y_table
            )

    def finalize_fit(self):
        """
        Finalizes ridge regression computation and obtains coefficients
        from the current `_partial_result`.

        Parameters
        ----------
        queue : dpctl.SyclQueue
            If available, uses provided queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        with SyclQueueManager.manage_global_queue(self._queue):
            result = self.finalize_train(self._params, self._partial_result)

        self._onedal_model = result.model

        packed_coefficients = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:].squeeze(),
            packed_coefficients[:, 0].squeeze(),
        )

        self._queue = None

        return self
