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

from ...covariance import (
    IncrementalEmpiricalCovariance as base_IncrementalEmpiricalCovariance,
)
from ...datatypes import to_table
from ...utils import _check_array
from .._base import BaseEstimatorSPMD


class IncrementalEmpiricalCovariance(
    BaseEstimatorSPMD, base_IncrementalEmpiricalCovariance
):
    def _reset(self):
        self._need_to_finalize = False
        self._partial_result = super(
            base_IncrementalEmpiricalCovariance, self
        )._get_backend("covariance", None, "partial_compute_result")

    def partial_fit(self, X, y=None, queue=None):
        """
        Computes partial data for the covariance matrix
        from data batch X and saves it to `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

        self._queue = queue

        policy = super(base_IncrementalEmpiricalCovariance, self)._get_policy(queue, X)

        X_table = to_table(X, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype

        params = self._get_onedal_params(self._dtype)
        self._partial_result = super(
            base_IncrementalEmpiricalCovariance, self
        )._get_backend(
            "covariance",
            None,
            "partial_compute",
            policy,
            params,
            self._partial_result,
            X_table,
        )
        self._need_to_finalize = True
