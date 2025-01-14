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

from ..datatypes import from_table, to_table
from ..utils import _check_array
from .basic_statistics import BaseBasicStatistics


class IncrementalBasicStatistics(BaseBasicStatistics):
    """
    Incremental estimator for basic statistics based on oneDAL implementation.
    Allows to compute basic statistics if data are splitted into batches.
    Parameters
    ----------
    result_options: string or list, default='all'
        List of statistics to compute

    Attributes (are existing only if corresponding result option exists)
    ----------
        min : ndarray of shape (n_features,)
            Minimum of each feature over all samples.

        max : ndarray of shape (n_features,)
            Maximum of each feature over all samples.

        sum : ndarray of shape (n_features,)
            Sum of each feature over all samples.

        mean : ndarray of shape (n_features,)
            Mean of each feature over all samples.

        variance : ndarray of shape (n_features,)
            Variance of each feature over all samples.

        variation : ndarray of shape (n_features,)
            Variation of each feature over all samples.

        sum_squares : ndarray of shape (n_features,)
            Sum of squares for each feature over all samples.

        standard_deviation : ndarray of shape (n_features,)
            Standard deviation of each feature over all samples.

        sum_squares_centered : ndarray of shape (n_features,)
            Centered sum of squares for each feature over all samples.

        second_order_raw_moment : ndarray of shape (n_features,)
            Second order moment of each feature over all samples.
    """

    def __init__(self, result_options="all"):
        super().__init__(result_options, algorithm="by_default")
        self._reset()

    def _reset(self):
        self._need_to_finalize = False
        # Not supported with spmd policy so IncrementalBasicStatistics must be specified
        self._partial_result = IncrementalBasicStatistics._get_backend(
            IncrementalBasicStatistics, "basic_statistics", None, "partial_compute_result"
        )

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.
        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)

        return data

    def partial_fit(self, X, weights=None, queue=None):
        """
        Computes partial data for basic statistics
        from data batch X and saves it to `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._queue = queue
        # Not supported with spmd policy so IncrementalBasicStatistics must be specified
        policy = IncrementalBasicStatistics._get_policy(
            IncrementalBasicStatistics, queue, X
        )

        X = _check_array(
            X, dtype=[np.float64, np.float32], ensure_2d=False, force_all_finite=False
        )
        if weights is not None:
            weights = _check_array(
                weights,
                dtype=[np.float64, np.float32],
                ensure_2d=False,
                force_all_finite=False,
            )

        if not hasattr(self, "_onedal_params"):
            dtype = get_dtype(X)
            self._onedal_params = self._get_onedal_params(False, dtype=dtype)

        X_table, weights_table = to_table(X, weights, queue=queue)
        # Not supported with spmd policy so IncrementalBasicStatistics must be specified
        self._partial_result = IncrementalBasicStatistics._get_backend(
            IncrementalBasicStatistics,
            "basic_statistics",
            None,
            "partial_compute",
            policy,
            self._onedal_params,
            self._partial_result,
            X_table,
            weights_table,
        )

        self._need_to_finalize = True
        return self

    def finalize_fit(self, queue=None):
        """
        Finalizes basic statistics computation and obtains result
        attributes from the current `_partial_result`.

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

            result = self._get_backend(
                "basic_statistics",
                None,
                "finalize_compute",
                policy,
                self._onedal_params,
                self._partial_result,
            )
            options = self._get_result_options(self.options).split("|")
            for opt in options:
                setattr(self, opt, from_table(getattr(result, opt)).ravel())

            self._need_to_finalize = False

        return self
