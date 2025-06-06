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
import warnings

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import check_array, gen_batches
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.linear_model import (
    IncrementalLinearRegression as onedal_IncrementalLinearRegression,
)
from sklearnex._config import get_config

from ..utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval

from onedal.common.hyperparameters import get_hyperparameters

from .._device_offload import dispatch, wrap_output_data
from .._utils import (
    PatchingConditionsChain,
    _add_inc_serialization_note,
    register_hyperparameters,
)
from ..base import oneDALEstimator


@register_hyperparameters(
    {
        "fit": get_hyperparameters("linear_regression", "train"),
        "partial_fit": get_hyperparameters("linear_regression", "train"),
    }
)
@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "predict", "score", "_onedal_finalize_fit"]
)
class IncrementalLinearRegression(
    MultiOutputMixin, RegressorMixin, oneDALEstimator, BaseEstimator
):
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

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "fit_intercept": ["boolean"],
            "copy_X": ["boolean"],
            "n_jobs": [Interval(numbers.Integral, -1, None, closed="left"), None],
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        }

    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, batch_size=None):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def _onedal_predict(self, X, queue=None):
        if get_config()["use_raw_input"] is False:
            if sklearn_check_version("1.2"):
                self._validate_params()

            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
                reset=False,
            )

        assert hasattr(self, "_onedal_estimator")
        if self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    def _onedal_partial_fit(self, X, y, check_input=True, queue=None):
        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        if sklearn_check_version("1.2"):
            self._validate_params()

        use_raw_input = get_config().get("use_raw_input", False) is True
        # never check input when using raw input
        check_input &= use_raw_input is False
        if check_input:
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[np.float64, np.float32],
                reset=first_pass,
                copy=self.copy_X,
                multi_output=True,
                ensure_all_finite=False,
            )

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
            self.n_features_in_ = X.shape[1]
        else:
            self.n_samples_seen_ += X.shape[0]
        onedal_params = {"fit_intercept": self.fit_intercept, "copy_X": self.copy_X}
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_linear(**onedal_params)
        self._onedal_estimator.partial_fit(X, y, queue=queue)
        self._need_to_finalize = True

    if daal_check_version((2025, "P", 200)):

        def _onedal_validate_underdetermined(self, n_samples, n_features):
            pass

    else:

        def _onedal_validate_underdetermined(self, n_samples, n_features):
            is_underdetermined = n_samples < n_features + int(self.fit_intercept)
            if is_underdetermined:
                raise ValueError("Not enough samples for oneDAL")

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_validate_underdetermined(self.n_samples_seen_, self.n_features_in_)
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

    def _onedal_fit(self, X, y, queue=None):
        if get_config()["use_raw_input"] is False:
            if sklearn_check_version("1.2"):
                self._validate_params()
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
                multi_output=True,
                ensure_2d=True,
            )

        n_samples, n_features = X.shape

        self._onedal_validate_underdetermined(n_samples, n_features)

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        for batch in gen_batches(n_samples, self.batch_size_):
            X_batch, y_batch = X[batch], y[batch]
            self._onedal_partial_fit(X_batch, y_batch, check_input=False, queue=queue)

        # finite check occurs on onedal side
        self.n_features_in_ = n_features

        if n_samples == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        self._onedal_finalize_fit()
        return self

    @property
    def intercept_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()

            return self._onedal_estimator.intercept_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'intercept_'"
            )

    @intercept_.setter
    def intercept_(self, value):
        self.__dict__["intercept_"] = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.intercept_ = value
            del self._onedal_estimator._onedal_model

    @property
    def coef_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()

            return self._onedal_estimator.coef_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coef_'"
            )

    @coef_.setter
    def coef_(self, value):
        self.__dict__["coef_"] = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.coef_ = value
            del self._onedal_estimator._onedal_model

    def partial_fit(self, X, y, check_input=True):
        """
        Incremental fit with X and y. X and y are processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where ``n_samples`` is the number of samples and
            ``n_targets`` is the number of targets.

        check_input : bool, default=True
            Run validate_data on X and y.

        Returns
        -------
        self : IncrementalLinearRegression
            Returns the instance itself.
        """

        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": None,
            },
            X,
            y,
            check_input=check_input,
        )
        return self

    def fit(self, X, y):
        """
        Fit the model with X and y, using minibatches of size ``batch_size``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. It is necessary for
            ``n_samples`` to be not less than ``n_features`` if ``fit_intercept``
            is False and not less than ``n_features + 1`` if ``fit_intercept``
            is 'True'.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where ``n_samples`` is the number of samples and
            ``n_targets`` is the number of targets.

        Returns
        -------
        self : IncrementalLinearRegression
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
            y,
        )
        return self

    @wrap_output_data
    def predict(self, X, y=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": None,
            },
            X,
        )

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": None,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    score.__doc__ = _sklearn_LinearRegression.score.__doc__
    predict.__doc__ = _sklearn_LinearRegression.predict.__doc__
