# ==============================================================================
# Copyright contributors to the oneDAL project
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
from abc import ABC, abstractmethod
from threading import Lock

from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import gen_batches
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, is_sparse, sklearn_check_version
from onedal.datatypes import from_table
from onedal.utils.validation import _num_samples

from .._device_offload import dispatch, wrap_output_data
from .._utils import (
    PatchingConditionsChain,
    register_hyperparameters,
)
from ..base import oneDALEstimator
from ..utils._array_api import (
    _NUMPY_FITTED_ATTRS,
    base_move_estimator_to,
    enable_array_api,
)
from ..utils.validation import validate_data

if sklearn_check_version("1.9"):
    from sklearn.utils._array_api import (
        check_same_namespace,
        get_namespace_and_device,
        move_to,
    )

# Note: this is a global object that would make all instances
# of these estimators compete for the same lock. These are not
# serializable, so if assigned inside an object, it would
# prevent serialization. But maybe the class using it could
# define a custom serializer that would avoid serializing a
# lock and then re-create it from scratch upon deserialization.
_inclinear_finalize_lock = Lock()


# Adds common methods to create oneDAL objects after a fallback
@register_hyperparameters({"fit": ("linear_regression", "train")})
class _BaseLinearModel(ABC):
    @abstractmethod
    def _initialize_onedal_estimator(self, override_fit_intercept: bool = False) -> None:
        raise NotImplementedError()

    def _onedal_predict_supported(self, patching_status, method_name, *data):
        assert method_name in ["predict", "score"]
        assert len(data) <= 2

        n_samples = _num_samples(data[0])
        model_is_sparse = is_sparse(self.coef_) or is_sparse(self.intercept_)
        patching_status.and_conditions(
            [
                (n_samples > 0, "Number of samples is less than 1."),
                (not is_sparse(data[0]), "Sparse input is not supported."),
                (not model_is_sparse, "Sparse coefficients are not supported."),
            ]
        )

        return patching_status

    def _onedal_predict(self, X, queue=None):
        xp, _ = get_namespace(X)

        X = validate_data(
            self, X, accept_sparse=False, dtype=[xp.float64, xp.float32], reset=False
        )

        if sklearn_check_version("1.10"):
            if not len(_NUMPY_FITTED_ATTRS):
                check_same_namespace(X, self, attribute="coef_", method="predict")
        elif sklearn_check_version("1.9"):
            check_same_namespace(X, self, attribute="coef_", method="predict")

        if not hasattr(self, "_onedal_estimator"):
            self._initialize_onedal_estimator_from_coefs()

        res = self._onedal_estimator.predict(X, queue=queue)
        if res.shape[1] == 1 and self.coef_.ndim == 1:
            res = xp.reshape(res, (-1,))

        return res

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        if sklearn_check_version("1.9"):
            xp, _, device = get_namespace_and_device(X)
            y = move_to(y, xp=xp, device=device)
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    @property
    def coef_(self):
        return self._coef_

    @coef_.setter
    def coef_(self, value):
        self._coef_ = value
        if hasattr(self, "_onedal_estimator"):
            del self._onedal_estimator
        if hasattr(self._coef_, "shape") and len(self._coef_.shape) > 1:
            self._force_coef_2d = True
        else:
            self._force_coef_2d = False

    @coef_.deleter
    def coef_(self):
        del self._coef_

    @property
    def intercept_(self):
        return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value
        if hasattr(self, "_onedal_estimator"):
            del self._onedal_estimator
        if hasattr(self._intercept_, "shape") and len(self._intercept_.shape):
            self._force_intercept_1d = True
        else:
            self._force_intercept_1d = False

    @intercept_.deleter
    def intercept_(self):
        del self._intercept_

    def _get_fit_intercept(self, override_fit_intercept: bool = False) -> bool:
        if not override_fit_intercept:
            return self.fit_intercept
        else:
            if isinstance(self.intercept_, float):
                return self.intercept_ != 0.0
            else:
                xp, _ = get_namespace(self.coef_)
                return bool(xp.any(self.intercept_ != 0))

    def _initialize_onedal_estimator_from_coefs(self) -> None:
        xp, _ = get_namespace(self.coef_)
        self._initialize_onedal_estimator(override_fit_intercept=True)
        self._onedal_estimator._create_model(self.coef_, self.intercept_, xp)

        # Note: this reassignment might look redundant, but it ensures
        # that the packed array in the oneDAL estimator and the independent
        # arrays in the sklearnex estimator end up sharing memory.
        self._coef_ = self._onedal_estimator.coef_
        self._intercept_ = self._onedal_estimator.intercept_
        self._reshape_coefficients_from_onedal()

    def _reshape_coefficients_from_onedal(self):
        if self._coef_.shape[0] == 1:
            self._coef_ = self._coef_[0, ...]  # set to 1d
            self._intercept_ = self._intercept_[0]  # set 1d to scalar

        if not len(self._coef_.shape):
            xp, _ = get_namespace(self._coef_)
            self._coef_ = xp.reshape(self._coef_, (-1,))

        if len(self._coef_.shape) < 2 and getattr(self, "_force_coef_2d", False):
            xp, _ = get_namespace(self._coef_)
            self._coef_ = xp.reshape(self._coef_, (1, -1))

        if not len(self._intercept_.shape) and getattr(
            self, "_force_intercept_1d", False
        ):
            xp, _ = get_namespace(self._coef_)
            self._intercept_ = xp.reshape(self._intercept_, (-1,))

    def __sklearn_array_api_convert__(self, converter):
        out, is_fitted, is_supported_device = base_move_estimator_to(
            self, converter, ["_onedal_estimator"], "_coef_"
        )
        if is_fitted and is_supported_device:
            out._initialize_onedal_estimator_from_coefs()
        return out


@enable_array_api
@register_hyperparameters(
    {
        "fit": ("linear_regression", "train"),
        "partial_fit": ("linear_regression", "train"),
    }
)
@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "predict", "score", "_onedal_finalize_fit"]
)
class _BaseIncrementalLinearModel(
    ABC, MultiOutputMixin, RegressorMixin, oneDALEstimator, BaseEstimator
):
    # Note: this is needed in order for the 'control_n_jobs' decorator to work
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    @abstractmethod
    def _initialize_empty_onedal_estimator(self, fit_intercept=None, alpha=None) -> None:
        raise NotImplementedError()

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.linear_model.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def _onedal_predict(self, X, queue=None):
        xp, _ = get_namespace(X)

        self._validate_params()

        X = validate_data(
            self,
            X,
            dtype=[xp.float64, xp.float32],
            copy=self.copy_X,
            reset=False,
        )

        if sklearn_check_version("1.10"):
            if not len(_NUMPY_FITTED_ATTRS):
                check_same_namespace(X, self, attribute="coef_", method="predict")
        elif sklearn_check_version("1.9"):
            check_same_namespace(X, self, attribute="coef_", method="predict")

        assert hasattr(self, "_onedal_estimator")

        if self._need_to_finalize:
            self._onedal_finalize_fit()
        # This can happen after manually reassigning attributes.
        # Should not be done directly when reassigning, since a
        # user might reassign the attributes to an different namespace
        # and leave things in an inconsistent state until assignments
        # are done and 'predict' is called.
        elif not hasattr(self._onedal_estimator, "_onedal_model"):
            xp, _ = get_namespace(self._coef_, self._intercept_)
            self._onedal_estimator._create_prediction_only_model(
                self._coef_, self._intercept_, xp
            )
            self._assign_coefficients_from_onedal_estimator()
        res = self._onedal_estimator.predict(X, queue=queue)

        if res.shape[1] == 1 and self.coef_.ndim == 1:
            res = xp.reshape(res, (-1,))
        return res

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        if sklearn_check_version("1.9"):
            xp, _, device_ = get_namespace_and_device(X)
            y = move_to(y, xp=xp, device=device_)
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    def _onedal_partial_fit(self, X, y, check_input=True, queue=None):
        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        if check_input:
            if sklearn_check_version("1.9"):
                xp, _ = get_namespace(X)
            else:
                xp, _ = get_namespace(X, y)
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[xp.float64, xp.float32],
                reset=first_pass,
                copy=self.copy_X,
                multi_output=True,
                y_numeric=True,
            )

            if not first_pass:
                expect_y_2d = getattr(self, "_force_coef_2d", False)
                if len(y.shape) < 2 and expect_y_2d:
                    raise ValueError(
                        "Attempting `partial_fit` on a single-dimensional 'y' on"
                        " a multi-dimensional estimator. Call `.fit()` again with"
                        " the new dimension of 'y' to reset expected shapes."
                    )
                elif len(y.shape) > 1 and not expect_y_2d:
                    raise ValueError(
                        "Attempting `partial_fit` on a multi-dimensional 'y' on"
                        " a single-dimensional estimator. Call `.fit() again with"
                        " the new dimension of 'y' to reset expected shapes."
                    )
            else:
                self._force_coef_2d = len(y.shape) > 1

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
            self.n_features_in_ = X.shape[1]
        else:
            self.n_samples_seen_ += X.shape[0]
        if not hasattr(self, "_onedal_estimator"):
            self._initialize_empty_onedal_estimator()
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
        with _inclinear_finalize_lock:
            self._onedal_estimator.finalize_fit()

            self.n_features_in_ = self._onedal_estimator.n_features_in_
            self._assign_coefficients_from_onedal_estimator()

            self._need_to_finalize = False

    def _assign_coefficients_from_onedal_estimator(self) -> None:
        self._coef_ = self._onedal_estimator.coef_
        self._intercept_ = self._onedal_estimator.intercept_

        if self._coef_.shape[0] == 1:
            self._coef_ = self._coef_[0, ...]  # set to 1d
            self._intercept_ = self._intercept_[0]  # set 1d to scalar

        if not len(self._coef_.shape):
            xp, _ = get_namespace(self._coef_)
            self._coef_ = xp.reshape(self._coef_, (-1,))

        if len(self._coef_.shape) < 2 and getattr(self, "_force_coef_2d", False):
            xp, _ = get_namespace(self._coef_)
            self._intercept_ = xp.reshape(self._intercept_, (-1,))
            self._coef_ = xp.reshape(self._coef_, (self._intercept_.shape[0], -1))

    def _onedal_fit(self, X, y, queue=None):

        if sklearn_check_version("1.9"):
            xp, _ = get_namespace(X)
        else:
            xp, _ = get_namespace(X, y)

        X, y = validate_data(
            self,
            X,
            y,
            dtype=[xp.float64, xp.float32],
            copy=self.copy_X,
            multi_output=True,
            y_numeric=True,
        )
        self._force_coef_2d = len(y.shape) > 1

        n_samples, n_features = X.shape

        self._onedal_validate_underdetermined(n_samples, n_features)

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            del self._onedal_estimator

        for batch in gen_batches(n_samples, self.batch_size_):
            X_batch, y_batch = X[batch, ...], y[batch, ...]
            self._onedal_partial_fit(X_batch, y_batch, check_input=False, queue=queue)

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
        self : Self
            Returns the instance itself.
        """
        if check_input:
            self._validate_params()

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
        self : Self
            Returns the instance itself.
        """
        self._validate_params()

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

    @property
    def coef_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._coef_

    @coef_.setter
    def coef_(self, value):
        if hasattr(self, "_onedal_estimator") and hasattr(
            self._onedal_estimator, "_onedal_model"
        ):
            del self._onedal_estimator._onedal_model
        self._coef_ = value
        self._need_to_finalize = False

    @coef_.deleter
    def coef_(self):
        del self._coef_

    @property
    def intercept_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        if hasattr(self, "_onedal_estimator"):
            if not self._onedal_estimator.fit_intercept:
                if isinstance(value, (float, int)):
                    new_fit_intercept = value != 0
                else:
                    xp, _ = get_namespace(value)
                    new_fit_intercept = xp.any(value != 0)
                if new_fit_intercept:
                    raise ValueError(
                        "Cannot set intercept_ on incremental estimator fitted without intercept."
                    )
        if hasattr(self, "_onedal_estimator") and hasattr(
            self._onedal_estimator, "_onedal_model"
        ):
            del self._onedal_estimator._onedal_model
        self._intercept_ = value
        self._need_to_finalize = False

    @intercept_.deleter
    def intercept_(self):
        del self._intercept_

    def __sklearn_array_api_convert__(self, converter):
        if getattr(self, "_need_to_finalize", False):
            self._onedal_finalize_fit()
        out, is_fitted, is_supported_device = base_move_estimator_to(
            self, converter, ["_onedal_estimator"], "_coef_"
        )
        if not is_fitted:
            return out
        # Note: the logic here should change if scikit-learn implements
        # their own version of this estimator.
        if not is_supported_device:
            raise ValueError(
                "Cannot move non-scikit-learn estimator to unsupported namespace/device"
            )
        coef = out._coef_
        intercept = out._intercept_
        xp, _ = get_namespace(coef, intercept)
        xtx = converter(from_table(self._onedal_estimator._partial_result.partial_xtx))
        xty = converter(from_table(self._onedal_estimator._partial_result.partial_xty))
        out._initialize_empty_onedal_estimator(
            fit_intercept=self._onedal_estimator.fit_intercept,
            alpha=self._onedal_estimator.alpha,
        )
        out._onedal_estimator._create_model(
            coef, intercept, xp, xtx, xty, self._onedal_estimator.n_features_in_
        )
        out._onedal_finalize_fit()
        return out

    score.__doc__ = _sklearn_LinearRegression.score.__doc__
    predict.__doc__ = _sklearn_LinearRegression.predict.__doc__
