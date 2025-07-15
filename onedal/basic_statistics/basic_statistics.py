# ==============================================================================
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
# ==============================================================================

from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table
from ..utils.validation import _is_csr


class BasicStatistics:
    """Low order moments oneDAL estimator.

    Calculate basic statistics for data.

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
        self.options = result_options
        self.algorithm = algorithm

    @bind_default_backend("basic_statistics")
    def compute(self, params, data_table, weights_table): ...

    @staticmethod
    def get_all_result_options():
        return [
            "min",
            "max",
            "sum",
            "mean",
            "variance",
            "variation",
            "sum_squares",
            "standard_deviation",
            "sum_squares_centered",
            "second_order_raw_moment",
        ]

    @property
    def options(self):
        if self._options == ["all"]:
            return self.get_all_result_options()
        return self._options

    @options.setter
    def options(self, opts):
        # options always to be an iterable
        self._options = opts.split("|") if isinstance(opts, str) else opts

    def _get_onedal_params(self, is_csr, dtype=None):
        return {
            "fptype": dtype,
            "method": "sparse" if is_csr else self.algorithm,
            "result_option": "|".join(self.options),
        }

    @supports_queue
    def fit(self, X, sample_weight=None, queue=None):
        """Generate statistics.

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

        is_csr = _is_csr(X)

        is_single_dim = X.ndim == 1
        X_table, sample_weight_table = to_table(X, sample_weight, queue=queue)

        result = self._compute_raw(X_table, sample_weight_table, X_table.dtype, is_csr)

        for opt in self.options:
            value = from_table(getattr(result, opt), like=X)[0]  # 2D table [1, n]
            setattr(self, opt + "_", value[0] if is_single_dim else value)

        return self

    def _compute_raw(self, data_table, weights_table, dtype=None, is_csr=False):
        params = self._get_onedal_params(is_csr, dtype)
        return self.compute(params, data_table, weights_table)
