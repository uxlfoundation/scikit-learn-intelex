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

import numbers

from sklearn.utils._param_validation import Interval

from onedal.linear_model import IncrementalRidge as onedal_IncrementalRidge

from .._utils import _add_inc_serialization_note
from ._base_linear_model import _BaseIncrementalLinearModel


class IncrementalRidge(_BaseIncrementalLinearModel):
    """
    Incremental estimator for Ridge Regression.

    Allows to train Ridge Regression if data is split into batches.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation.

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the ridge regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
        It should be not less than `n_features_in_` if `fit_intercept`
        is False and not less than `n_features_in_` + 1 if `fit_intercept`
        is True to obtain regression coefficients.

    batch_size_ : int
        Inferred batch size from ``batch_size``.

    Notes
    -----
    Sparse data formats are not supported. Input dtype must be ``float32`` or ``float64``.

    %incremental_serialization_note%
    """

    __doc__ = _add_inc_serialization_note(__doc__)

    _onedal_incremental_ridge = staticmethod(onedal_IncrementalRidge)

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "copy_X": ["boolean"],
        "n_jobs": [numbers.Integral, None],
        "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self, fit_intercept=True, alpha=1.0, copy_X=True, n_jobs=None, batch_size=None
    ):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.copy_X = copy_X
        self.batch_size = batch_size
        super().__init__(n_jobs)

    def _initialize_empty_onedal_estimator(self) -> None:
        onedal_params = {
            "fit_intercept": self.fit_intercept,
            "alpha": self.alpha,
            "copy_X": self.copy_X,
        }
        self._onedal_estimator = self._onedal_incremental_ridge(**onedal_params)
