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

from ...common.hyperparameters import get_hyperparameters
from ...datatypes import to_table
from ...linear_model import (
    IncrementalLinearRegression as base_IncrementalLinearRegression,
)
from ...utils import _check_X_y, _num_features
from .._base import BaseEstimatorSPMD


class IncrementalLinearRegression(BaseEstimatorSPMD, base_IncrementalLinearRegression):
    """
    Distributed incremental Linear Regression oneDAL implementation.

    API is the same as for `onedal.linear_model.IncrementalLinearRegression`.
    """

    def _reset(self):
        self._need_to_finalize = False
        self._partial_result = super(base_IncrementalLinearRegression, self)._get_backend(
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
        module = super(base_IncrementalLinearRegression, self)._get_backend(
            "linear_model", "regression"
        )

        self._queue = queue
        policy = super(base_IncrementalLinearRegression, self)._get_policy(queue, X)

        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32], accept_2d_y=True, force_all_finite=False
        )

        X_table, y_table = to_table(X, y, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(self._dtype)

        y = np.asarray(y, dtype=self._dtype)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

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
