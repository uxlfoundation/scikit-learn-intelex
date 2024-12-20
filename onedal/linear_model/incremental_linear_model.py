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

from .._config import _get_config
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils import _check_X_y, _num_features
from ..utils._array_api import _get_sycl_namespace
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
        self._reset()

    def _reset(self):
        self._partial_result = self._get_backend(
            "linear_model", "regression", "partial_train_result"
        )

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
        if not hasattr(self, "_params"):
            self._params = self._get_onedal_params(X.dtype)

        if _get_config().get("use_raw_input") is False:
            X, y = _check_X_y(
                X,
                y,
                dtype=[np.float64, np.float32],
                accept_2d_y=True,
                force_all_finite=False,
            )
            y = np.asarray(y, dtype=X.dtype)

        module = self._get_backend("linear_model", "regression")

        policy = self._get_policy(queue, X)
        queue = self._queue = getattr(policy, "_queue", None)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)

        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            self._partial_result = module.partial_train(
                policy,
                self._params,
                hparams.backend,
                self._partial_result,
                X_table,
                y_table,
            )
        else:
            self._partial_result = module.partial_train(
                policy, self._params, self._partial_result, X_table, y_table
            )

    def finalize_fit(self, queue=None):
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
        if queue is not None:
            policy = self._get_policy(queue)
        else:
            policy = self._get_policy(self._queue)

        module = self._get_backend("linear_model", "regression")
        hparams = get_hyperparameters("linear_regression", "train")
        if hparams is not None and not hparams.is_default:
            result = module.finalize_train(
                policy, self._params, hparams.backend, self._partial_result
            )
        else:
            result = module.finalize_train(policy, self._params, self._partial_result)

        self._onedal_model = result.model

        packed_coefficients = from_table(
            result.model.packed_coefficients, sycl_queue=self._queue
        )
        self.coef_, self.intercept_ = (
            packed_coefficients[:, 1:].squeeze(),
            packed_coefficients[:, 0].squeeze(),
        )

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
        module = self._get_backend("linear_model", "regression")
        super().__init__(
            fit_intercept=fit_intercept, alpha=alpha, copy_X=copy_X, algorithm=algorithm
        )
        self._partial_result = module.partial_train_result()

    def _reset(self):
        module = self._get_backend("linear_model", "regression")
        self._partial_result = module.partial_train_result()

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
        if not hasattr(self, "_params"):
            self._params = self._get_onedal_params(X.dtype)

        module = self._get_backend("linear_model", "regression")

        self._sua_iface, self._xp, _ = _get_sycl_namespace(X)
        use_raw_input = _get_config().get("use_raw_input") is True
        if use_raw_input and self._sua_iface is not None:
            queue = X.sycl_queue

        self._queue = queue
        policy = self._get_policy(queue, X)

        if not use_raw_input:
            X, y = _check_X_y(
                X,
                y,
                dtype=[np.float64, np.float32],
                accept_2d_y=True,
                force_all_finite=False,
            )
            y = np.asarray(y, dtype=X.dtype)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)

        self._partial_result = module.partial_train(
            policy, self._params, self._partial_result, X_table, y_table
        )

    def finalize_fit(self, queue=None):
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
        module = self._get_backend("linear_model", "regression")
        if queue is not None:
            policy = self._get_policy(queue)
        else:
            policy = self._get_policy(self._queue)
        result = module.finalize_train(policy, self._params, self._partial_result)

        self._onedal_model = result.model

        if _get_config().get("use_raw_input") is True:
            packed_coefficients = from_table(
                result.model.packed_coefficients,
                sua_iface=self._sua_iface,
                sycl_queue=self._queue,
                xp=self._xp,
            )
            self.coef_, self.intercept_ = (
                self._xp.squeeze(packed_coefficients[:, 1:]),
                self._xp.squeeze(packed_coefficients[:, 0]),
            )
        else:
            packed_coefficients = from_table(result.model.packed_coefficients)
            self.coef_, self.intercept_ = (
                packed_coefficients[:, 1:].squeeze(),
                packed_coefficients[:, 0].squeeze(),
            )

        return self
