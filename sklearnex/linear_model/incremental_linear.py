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
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.linear_model import (
    IncrementalLinearRegression as onedal_IncrementalLinearRegression,
)

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = BaseEstimator._validate_data

from onedal.common.hyperparameters import get_hyperparameters

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, register_hyperparameters


@register_hyperparameters(
    {
        "fit": get_hyperparameters("linear_regression", "train"),
        "partial_fit": get_hyperparameters("linear_regression", "train"),
    }
)
@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "predict", "score", "_onedal_finalize_fit"]
)
class IncrementalLinearRegression(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    Incremental estimator for linear regression.
    Allows to train linear regression if data are splitted into batches.

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
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

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

    n_features_in_ : int
        Number of features seen during :term:`fit` `partial_fit`.

    """

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
        if sklearn_check_version("1.2"):
            self._validate_params()

        if sklearn_check_version("1.0"):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
                reset=False,
            )
        else:
            X = check_array(
                X,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
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

        if check_input:
            if sklearn_check_version("1.0"):
                X, y = validate_data(
                    self,
                    X,
                    y,
                    dtype=[np.float64, np.float32],
                    reset=first_pass,
                    copy=self.copy_X,
                    multi_output=True,
                    force_all_finite=False,
                )
            else:
                X = check_array(
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy_X,
                    force_all_finite=False,
                )
                y = check_array(
                    y,
                    dtype=[np.float64, np.float32],
                    copy=False,
                    ensure_2d=False,
                    force_all_finite=False,
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

    def _onedal_finalize_fit(self, queue=None):
        assert hasattr(self, "_onedal_estimator")
        is_underdetermined = self.n_samples_seen_ < self.n_features_in_ + int(
            self.fit_intercept
        )
        if is_underdetermined:
            raise ValueError("Not enough samples to finalize")
        self._onedal_estimator.finalize_fit(queue=queue)
        self._need_to_finalize = False

    def _onedal_fit(self, X, y, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        if sklearn_check_version("1.0"):
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
                multi_output=True,
                ensure_2d=True,
            )
        else:
            X = check_array(
                X,
                dtype=[np.float64, np.float32],
                copy=self.copy_X,
            )
            y = check_array(
                y,
                dtype=[np.float64, np.float32],
                copy=False,
                ensure_2d=False,
            )

        n_samples, n_features = X.shape

        is_underdetermined = n_samples < n_features + int(self.fit_intercept)
        if is_underdetermined:
            raise ValueError("Not enough samples to run oneDAL backend")

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

        if sklearn_check_version("1.2"):
            self._validate_params()

        # finite check occurs on onedal side
        self.n_features_in_ = n_features

        if n_samples == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        self._onedal_finalize_fit(queue=queue)
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
        Incremental fit linear model with X and y. All of X and y is
        processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where `n_samples` is the number of samples and
            `n_targets` is the number of targets.

        Returns
        -------
        self : object
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
        Fit the model with X and y, using minibatches of size batch_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features. It is necessary for
            `n_samples` to be not less than `n_features` if `fit_intercept`
            is False and not less than `n_features` + 1 if `fit_intercept`
            is True

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where `n_samples` is the number of samples and
            `n_targets` is the number of targets.

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
            y,
        )
        return self

    @wrap_output_data
    def predict(self, X, y=None):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples, n_targets)
            Returns predicted values.
        """
        if not hasattr(self, "coef_"):
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' or 'partial_fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": self.__class__.__name__})

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
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """
        if not hasattr(self, "coef_"):
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' or 'partial_fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": self.__class__.__name__})

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
