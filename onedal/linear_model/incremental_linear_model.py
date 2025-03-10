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
        self._need_to_finalize = False
        # Not supported with spmd policy so IncrementalLinearRegression must be specified
        self._partial_result = IncrementalLinearRegression._get_backend(
            IncrementalLinearRegression,
            "linear_model",
            "regression",
            "partial_train_result",
        )

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.

        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)

        return data

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
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, _, _ = _get_sycl_namespace(X)

        if use_raw_input and sua_iface:
            queue = X.sycl_queue
        if not use_raw_input:
            X, y = _check_X_y(
                X,
                y,
                dtype=[np.float64, np.float32],
                accept_2d_y=True,
                force_all_finite=False,
            )
            y = np.asarray(y, dtype=X.dtype)

        if not hasattr(self, "_params"):
            self._params = self._get_onedal_params(X.dtype)

        # Not supported with spmd policy so IncrementalLinearRegression must be specified
        module = IncrementalLinearRegression._get_backend(
            IncrementalLinearRegression, "linear_model", "regression"
        )

        self._queue = queue
        # Not supported with spmd policy so IncrementalLinearRegression must be specified
        policy = IncrementalLinearRegression._get_policy(
            IncrementalLinearRegression, queue, X
        )

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

        self._need_to_finalize = True
        return self

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

        if self._need_to_finalize:
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
            self._need_to_finalize = False

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
        self._reset()

    def _reset(self):
        module = self._get_backend("linear_model", "regression")
        self._partial_result = module.partial_train_result()
        self._need_to_finalize = False

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.

        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)

        return data

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
        module = self._get_backend("linear_model", "regression")

        self._queue = queue
        policy = self._get_policy(queue, X)

        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32], accept_2d_y=True, force_all_finite=False
        )
        y = np.asarray(y, dtype=X.dtype)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(self._dtype)

        self._partial_result = module.partial_train(
            policy, self._params, self._partial_result, X_table, y_table
        )

        self._need_to_finalize = True
        return self

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

        if self._need_to_finalize:
            module = self._get_backend("linear_model", "regression")
            if queue is not None:
                policy = self._get_policy(queue)
            else:
                policy = self._get_policy(self._queue)
            result = module.finalize_train(policy, self._params, self._partial_result)

            self._onedal_model = result.model

            packed_coefficients = from_table(result.model.packed_coefficients)
            self.coef_, self.intercept_ = (
                packed_coefficients[:, 1:].squeeze(),
                packed_coefficients[:, 0].squeeze(),
            )

            self._need_to_finalize = False

        return self
