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

from .._config import get_config
from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from .basic_statistics import BasicStatistics


class IncrementalBasicStatistics(BasicStatistics):
    """Incremental oneDAL low order moments estimator.

    Calculate basic statistics for data split into batches.

    Parameters
    ----------
    result_options : str or list, default=str('all')
        List of statistics to compute.

    algorithm : str, default=str('by_default')
        Method for statistics computation.

    Attributes
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

    Notes
    -----
        Attributes are populated only for corresponding result options.
    """

    def __init__(self, result_options="all", algorithm="by_default"):
        super().__init__(result_options, algorithm)
        self._reset()
        self._queue = None

    @bind_default_backend("basic_statistics")
    def partial_compute_result(self): ...

    @bind_default_backend("basic_statistics")
    def partial_compute(self, *args, **kwargs): ...

    @bind_default_backend("basic_statistics")
    def finalize_compute(self, *args, **kwargs): ...

    def _reset(self):
        self._need_to_finalize = False
        self._queue = None
        # get the _partial_result pointer from backend
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
    def partial_fit(self, X, sample_weight=None, queue=None):
        """Generate partial statistics from batch data in `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, _, _ = _get_sycl_namespace(X)

        # All data should use the same sycl queue
        if use_raw_input and sua_iface:
            queue = X.sycl_queue

        self._queue = queue
        X_table, sample_weight_table = to_table(X, sample_weight, queue=queue)

        if not hasattr(self, "_onedal_params"):
            self._onedal_params = self._get_onedal_params(False, dtype=X.dtype)

        self._partial_result = self.partial_compute(
            self._onedal_params, self._partial_result, X_table, sample_weight_table
        )

        self._need_to_finalize = True
        self._queue = queue

    def finalize_fit(self):
        """Finalize statistics from the current `_partial_result`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self._need_to_finalize:
            with QM.manage_global_queue(self._queue):
                result = self.finalize_compute(self._onedal_params, self._partial_result)

            for opt in self.options:
                setattr(self, opt, from_table(getattr(result, opt))[0])

            self._need_to_finalize = False

        return self
