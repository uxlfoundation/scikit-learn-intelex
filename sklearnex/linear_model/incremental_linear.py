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

from onedal.linear_model import (
    IncrementalLinearRegression as onedal_IncrementalLinearRegression,
)

from .._utils import (
    _add_inc_serialization_note,
)
from ._base_linear_model import _BaseIncrementalLinearModel


class IncrementalLinearRegression(_BaseIncrementalLinearModel):
    """
    Incremental Ordinary least squares Linear Regression.

    Trains a linear regression model, allows for computation if the data
    are split into batches. The user can use the ``partial_fit`` method
    to provide a single batch of data or use the ``fit`` method to provide
    the entire dataset.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation.

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to ``fit``, but increments across ``partial_fit`` calls.
        It should be not less than `n_features_in_` if `fit_intercept`
        is False and not less than `n_features_in_` + 1 if `fit_intercept`
        is True to obtain regression coefficients.

    batch_size_ : int
        Inferred batch size from ``batch_size``.

    n_features_in_ : int
        Number of features seen during ``fit`` or ``partial_fit``.

    Notes
    -----
    Sparse data formats are not supported. Input dtype must be ``float32`` or ``float64``.

    %incremental_serialization_note%

    Examples
    --------
    >>> import numpy as np
    >>> from sklearnex.linear_model import IncrementalLinearRegression
    >>> inclr = IncrementalLinearRegression(batch_size=2)
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 10]])
    >>> y = np.array([1.5, 3.5, 5.5, 8.5])
    >>> inclr.partial_fit(X[:2], y[:2])
    >>> inclr.partial_fit(X[2:], y[2:])
    >>> inclr.coef_
    np.array([0.5., 0.5.])
    >>> inclr.intercept_
    np.array(0.)
    >>> inclr.fit(X)
    >>> inclr.coef_
    np.array([0.5., 0.5.])
    >>> inclr.intercept_
    np.array(0.)
    """

    __doc__ = _add_inc_serialization_note(__doc__)

    _onedal_incremental_linear = staticmethod(onedal_IncrementalLinearRegression)

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "n_jobs": [numbers.Integral, None],
        "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, batch_size=None):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.batch_size = batch_size
        super().__init__(n_jobs)

    def _initialize_empty_onedal_estimator(self) -> None:
        onedal_params = {"fit_intercept": self.fit_intercept, "copy_X": self.copy_X}
        self._onedal_estimator = self._onedal_incremental_linear(**onedal_params)
