# ===============================================================================
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
# ===============================================================================
import numpy as np

from daal4py.sklearn._utils import daal_check_version

from .._config import _get_config
from ..datatypes import from_table, to_table
from ..utils import _check_array
from ..utils._array_api import _get_sycl_namespace
from .covariance import BaseEmpiricalCovariance


class IncrementalEmpiricalCovariance(BaseEmpiricalCovariance):
    """
    Covariance estimator based on oneDAL implementation.

    Computes sample covariance matrix.

    Parameters
    ----------
    method : string, default="dense"
        Specifies computation method. Available methods: "dense".

    bias: bool, default=False
        If True biased estimation of covariance is computed which equals to
        the unbiased one multiplied by (n_samples - 1) / n_samples.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e., the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    """

    def __init__(self, method="dense", bias=False, assume_centered=False):
        super().__init__(method, bias, assume_centered)
        self._reset()

    def _reset(self):
        self._need_to_finalize = False
        # Not supported with spmd policy so IncrementalEmpiricalCovariance must be specified
        self._partial_result = IncrementalEmpiricalCovariance._get_backend(
            IncrementalEmpiricalCovariance, "covariance", None, "partial_compute_result"
        )

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.

        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)

        return data

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
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, _, _ = _get_sycl_namespace(X)

        if use_raw_input and sua_iface:
            queue = X.sycl_queue
        if not use_raw_input:
            X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

        self._queue = queue

        # Not supported with spmd policy so IncrementalEmpiricalCovariance must be specified
        policy = IncrementalEmpiricalCovariance._get_policy(
            IncrementalEmpiricalCovariance, queue, X
        )

        X_table = to_table(X, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype

        params = self._get_onedal_params(self._dtype)
        # Not supported with spmd policy so IncrementalEmpiricalCovariance must be specified
        self._partial_result = IncrementalEmpiricalCovariance._get_backend(
            IncrementalEmpiricalCovariance,
            "covariance",
            None,
            "partial_compute",
            policy,
            params,
            self._partial_result,
            X_table,
        )
        self._need_to_finalize = True

    def finalize_fit(self, queue=None):
        """
        Finalizes covariance matrix and obtains `covariance_` and `location_`
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
            params = self._get_onedal_params(self._dtype)
            if queue is not None:
                policy = self._get_policy(queue)
            else:
                policy = self._get_policy(self._queue)

            result = self._get_backend(
                "covariance",
                None,
                "finalize_compute",
                policy,
                params,
                self._partial_result,
            )
            if daal_check_version((2024, "P", 1)) or (not self.bias):
                self.covariance_ = from_table(result.cov_matrix)
            else:
                n_rows = self._partial_result.partial_n_rows
                self.covariance_ = from_table(result.cov_matrix) * (n_rows - 1) / n_rows

            self.location_ = from_table(result.means).ravel()

            self._need_to_finalize = False

        return self
