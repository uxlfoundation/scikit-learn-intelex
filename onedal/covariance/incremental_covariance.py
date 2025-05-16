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
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from .._config import _get_config
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import _check_array
from .covariance import BaseEmpiricalCovariance


class IncrementalEmpiricalCovariance(BaseEmpiricalCovariance):
    """Covariance estimator based on oneDAL implementation.

    Computes sample covariance matrix.

    Parameters
    ----------
    method : str, default="dense"
        Specifies computation method. Available methods: "dense".

    bias : bool, default=False
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
        self._queue = None

    @bind_default_backend("covariance")
    def partial_compute(self, params, partial_result, X_table): ...

    @bind_default_backend("covariance")
    def partial_compute_result(self): ...

    @bind_default_backend("covariance")
    def finalize_compute(self, params, partial_result): ...

    def _reset(self):
        self._need_to_finalize = False
        self._queue = None
        self._partial_result = self.partial_compute_result()

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.

        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)

        return data

    @supports_queue
    def partial_fit(self, X, y=None, queue=None):
        """Generate partial covariance from batch data in `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        queue : SyclQueue or None, default=None
            If not None, use this queue for computation.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        use_raw_input = _get_config()["use_raw_input"] is True
        sua_iface, _, _ = _get_sycl_namespace(X)

        if use_raw_input and sua_iface:
            queue = X.sycl_queue
        if not use_raw_input:
            X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

        self._queue = queue
        X_table = to_table(X, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype

        params = self._get_onedal_params(self._dtype)
        self._partial_result = self.partial_compute(params, self._partial_result, X_table)
        self._need_to_finalize = True
        # store the queue for when we finalize
        self._queue = queue

    def finalize_fit(self):
        """Finalize covariance matrix from the current `_partial_result`.

        Results are stored in `location_` and `covariance_` attributes.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self._need_to_finalize:
            params = self._get_onedal_params(self._dtype)
            with QM.manage_global_queue(self._queue):
                result = self.finalize_compute(params, self._partial_result)

            if daal_check_version((2024, "P", 1)) or (not self.bias):
                self.covariance_ = from_table(result.cov_matrix)
            else:
                n_rows = self._partial_result.partial_n_rows
                self.covariance_ = from_table(result.cov_matrix) * (n_rows - 1) / n_rows

            self.location_ = from_table(result.means).ravel()

            self._need_to_finalize = False

        return self
