# ===============================================================================
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
# ===============================================================================
from abc import ABCMeta

import numpy as np

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils.validation import _check_array

from .._config import _get_config
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import _check_array


class BaseEmpiricalCovariance(metaclass=ABCMeta):
    def __init__(self, method="dense", bias=False, assume_centered=False):
        self.method = method
        self.bias = bias
        self.assume_centered = assume_centered

    @bind_default_backend("covariance")
    def compute(self, *args, **kwargs): ...

    def _get_onedal_params(self, dtype=np.float32):
        params = {
            "fptype": dtype,
            "method": self.method,
        }
        if daal_check_version((2024, "P", 1)):
            params["bias"] = self.bias
        if daal_check_version((2024, "P", 400)):
            params["assumeCentered"] = self.assume_centered

        return params


class EmpiricalCovariance(BaseEmpiricalCovariance):
    """Covariance estimator.

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

    @supports_queue
    def fit(self, X, y=None, queue=None):
        """Fit the sample covariance matrix of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples, and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

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
            X = _check_array(X, dtype=[np.float64, np.float32])
        X = to_table(X, queue=queue)

        params = self._get_onedal_params(X.dtype)
        hparams = get_hyperparameters("covariance", "compute")
        if hparams is not None and not hparams.is_default:
            result = self.compute(params, hparams.backend, X)
        else:
            result = self.compute(params, X)
        if daal_check_version((2024, "P", 1)) or (not self.bias):
            self.covariance_ = from_table(result.cov_matrix, sycl_queue=queue)
        else:
            self.covariance_ = (
                from_table(result.cov_matrix, sycl_queue=queue)
                * (X.shape[0] - 1)
                / X.shape[0]
            )

        self.location_ = from_table(result.means, sycl_queue=queue).ravel()

        return self
