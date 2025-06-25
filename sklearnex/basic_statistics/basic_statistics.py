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

import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.basic_statistics import BasicStatistics as onedal_BasicStatistics
from onedal.utils.validation import _is_csr

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import StrOptions


@control_n_jobs(decorated_methods=["fit"])
class BasicStatistics(oneDALEstimator, BaseEstimator):
    """
    Estimator for basic statistics.

    Compute low order moments and related statistics for given data.

    Parameters
    ----------
    result_options : str or list, default=str('all')
        Used to set statistics to calculate. Possible values are ``'min'``, ``'max'``, ``'sum'``, ``'mean'``, ``'variance'``,
        ``'variation'``, ``sum_squares'``, ``sum_squares_centered'``, ``'standard_deviation'``, ``'second_order_raw_moment'``
        or a list containing any of these values. If set to ``'all'`` then all possible statistics will be
        calculated.

    Attributes
    ----------
        min_ : ndarray of shape (n_features,)
            Minimum of each feature over all samples.
        max_ : ndarray of shape (n_features,)
            Maximum of each feature over all samples.
        sum_ : ndarray of shape (n_features,)
            Sum of each feature over all samples.
        mean_ : ndarray of shape (n_features,)
            Mean of each feature over all samples.
        variance_ : ndarray of shape (n_features,)
            Variance of each feature over all samples. Bessel's correction is used.
        variation_ : ndarray of shape (n_features,)
            Variation of each feature over all samples. Bessel's correction is used.
        sum_squares_ : ndarray of shape (n_features,)
            Sum of squares for each feature over all samples.
        standard_deviation_ : ndarray of shape (n_features,)
            Unbiased standard deviation of each feature over all samples. Bessel's correction is used.
        sum_squares_centered_ : ndarray of shape (n_features,)
            Centered sum of squares for each feature over all samples.
        second_order_raw_moment_ : ndarray of shape (n_features,)
            Second order moment of each feature over all samples.

    Notes
    -----
    Attribute exists only if corresponding result option has been provided.

    Names of attributes without the trailing underscore are
    supported currently but deprecated in 2025.1 and will be removed in 2026.0

    Some results can exhibit small variations due to
    floating point error accumulation and multithreading.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearnex.basic_statistics import BasicStatistics
    >>> bs = BasicStatistics(result_options=['sum', 'min', 'max'])
    >>> X = np.array([[1, 2], [3, 4]])
    >>> bs.fit(X)
    >>> bs.sum_
    np.array([4., 6.])
    >>> bs.min_
    np.array([1., 2.])
    """

    def __init__(self, result_options="all"):
        self.result_options = result_options

    _onedal_basic_statistics = staticmethod(onedal_BasicStatistics)

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "result_options": [
                StrOptions(
                    {
                        "all",
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
                    }
                ),
                list,
            ],
        }

    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")

        if self.result_options == "all":
            result_options = onedal_BasicStatistics.get_all_result_options()
        else:
            result_options = self.result_options

        if isinstance(result_options, str):
            setattr(
                self,
                result_options + "_",
                getattr(self._onedal_estimator, result_options),
            )
        elif isinstance(result_options, list):
            for option in result_options:
                setattr(self, option + "_", getattr(self._onedal_estimator, option))

    def __getattr__(self, attr):
        if self.result_options == "all":
            result_options = onedal_BasicStatistics.get_all_result_options()
        else:
            result_options = self.result_options
        is_deprecated_attr = (
            isinstance(result_options, str) and (attr == result_options)
        ) or (isinstance(result_options, list) and (attr in result_options))
        if is_deprecated_attr:
            warnings.warn(
                "Result attributes without a trailing underscore were deprecated in version 2025.1 and will be removed in 2026.0"
            )
            attr += "_"
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def _onedal_cpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearnex.basic_statistics.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearnex.basic_statistics.{self.__class__.__name__}.{method_name}"
        )
        X, sample_weight = data

        is_data_supported = not issparse(X) or (
            _is_csr(X) and daal_check_version((2025, "P", 200))
        )

        is_sample_weight_supported = sample_weight is None or not issparse(X)

        patching_status.and_conditions(
            [
                (
                    is_sample_weight_supported,
                    "Sample weights are not supported for CSR data format",
                ),
                (
                    is_data_supported,
                    "Supported data formats: Dense, CSR (oneDAL version >= 2025.2.0).",
                ),
            ]
        )
        return patching_status

    def _onedal_fit(self, X, sample_weight=None, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            ensure_2d=False,
            accept_sparse="csr",
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        onedal_params = {
            "result_options": self.result_options,
        }

        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_basic_statistics(**onedal_params)
        self._onedal_estimator.fit(X, sample_weight, queue=queue)
        self._save_attributes()
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1

    def fit(self, X, y=None, sample_weight=None):
        """Calculate statistics of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for compute, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights for compute weighted statistics, where ``n_samples`` is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": None,
            },
            X,
            sample_weight,
        )
        return self
