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
from sklearn.linear_model import Ridge as _sklearn_Ridge
from sklearn.metrics import r2_score
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_is_fitted, check_X_y

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn.utils.validation import sklearn_check_version

from ..utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval

from onedal.linear_model import IncrementalRidge as onedal_IncrementalRidge

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, _add_inc_serialization_note
from ..base import oneDALEstimator


@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "predict", "score", "_onedal_finalize_fit"]
)
class IncrementalRidge(MultiOutputMixin, RegressorMixin, oneDALEstimator, BaseEstimator):
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

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "fit_intercept": ["boolean"],
            "alpha": [Interval(numbers.Real, 0, None, closed="left")],
            "copy_X": ["boolean"],
            "n_jobs": [Interval(numbers.Integral, -1, None, closed="left"), None],
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        }

    def __init__(
        self, fit_intercept=True, alpha=1.0, copy_X=True, n_jobs=None, batch_size=None
    ):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
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

        X = validate_data(self, X, accept_sparse=False, reset=False)

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
        onedal_params = {
            "fit_intercept": self.fit_intercept,
            "alpha": self.alpha,
            "copy_X": self.copy_X,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_ridge(**onedal_params)
        self._onedal_estimator.partial_fit(X, y, queue=queue)
        self._need_to_finalize = True

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        is_underdetermined = self.n_samples_seen_ < self.n_features_in_ + int(
            self.fit_intercept
        )
        if is_underdetermined:
            raise ValueError("Not enough samples to finalize")
        self._onedal_estimator.finalize_fit()
        self._save_attributes()
        self._need_to_finalize = False

    def _onedal_fit(self, X, y, queue=None):
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

        self._onedal_finalize_fit()

        return self

    def partial_fit(self, X, y, check_input=True):
        """
        Incrementally fits with X and y. X and y are processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where `n_samples` is the number of samples and
            `n_targets` is the number of targets.

        check_input : bool, default=True
            Run validate_data on X and y.

        Returns
        -------
        self : IncrementalRidge
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
            Training data, where `n_samples` is the number of samples and
            ``n_features`` is the number of features. It is necessary for
            ``n_samples`` to be not less than ``n_features`` if ``fit_intercept``
            is False and not less than ``n_features`` + 1 if ``fit_intercept``
            is True.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values, where `n_samples` is the number of samples and
            `n_targets` is the number of targets.

        Returns
        -------
        self : IncrementalRidge
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
        check_is_fitted(
            self,
            msg=f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
        )

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
        check_is_fitted(
            self,
            msg=f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
        )

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

    score.__doc__ = _sklearn_Ridge.score.__doc__
    predict.__doc__ = _sklearn_Ridge.predict.__doc__

    @property
    def coef_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()

        return self._coef

    @coef_.setter
    def coef_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.coef_ = value
            # checking if the model is already fitted and if so, deleting the model
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._coef = value

    @property
    def intercept_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()

        return self._intercept

    @intercept_.setter
    def intercept_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.intercept_ = value
            # checking if the model is already fitted and if so, deleting the model
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._intercept = value

    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self._coef = self._onedal_estimator.coef_
        self._intercept = self._onedal_estimator.intercept_
