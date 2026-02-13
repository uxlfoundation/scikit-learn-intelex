# ==============================================================================
# Copyright Contributors to the oneDAL Project
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
from functools import wraps
from numbers import Real

import numpy as np
from scipy import sparse as sp
from sklearn.base import RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm._base import BaseLibSVM as _sklearn_BaseLibSVM
from sklearn.svm._base import BaseSVC as _sklearn_BaseSVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d

from daal4py.sklearn._utils import sklearn_check_version

from .._config import config_context, get_config
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import get_namespace
from ..utils.class_weight import _compute_class_weight
from ..utils.validation import _check_sample_weight, validate_data

if sklearn_check_version("1.6"):
    from sklearn.calibration import _fit_calibrator
    from sklearn.frozen import FrozenEstimator
    from sklearn.utils import indexable
    from sklearn.utils._response import _get_response_values
    from sklearn.utils.validation import check_is_fitted

    def _prefit_CalibratedClassifierCV_fit(self, X, y, **fit_params):
        # This is a stop-gap solution where the cv='prefit' of CalibratedClassifierCV
        # was removed and the single fold solution needs to be maintained. Discussion
        # of the mathematical and performance implications of this choice can be found
        # here: https://github.com/uxlfoundation/scikit-learn-intelex/pull/1879
        # This is distilled from the sklearn CalibratedClassifierCV for sklearn <1.8 for
        # use in sklearn > 1.8 to maintain performance.
        xp, _ = get_namespace(X, y)
        check_classification_targets(y)
        X, y = indexable(X, y)

        estimator = self._get_estimator()

        self.calibrated_classifiers_ = []
        check_is_fitted(self.estimator, attributes=["classes_"])
        self.classes_ = self.estimator.classes_

        predictions, _ = _get_response_values(
            estimator,
            X,
            response_method=["decision_function", "predict_proba"],
        )
        if predictions.ndim == 1:
            # Reshape binary output from `(n_samples,)` to `(n_samples, 1)`
            predictions = xp.reshape(predictions, (-1, 1))

        if sklearn_check_version("1.8"):
            xp, _ = get_namespace(X, y)
            calibrated_classifier = _fit_calibrator(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                xp,
            )
        else:
            calibrated_classifier = _fit_calibrator(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
            )
        self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self


class BaseSVM(oneDALEstimator):

    _onedal_factory = None

    @property
    def _dual_coef_(self):
        return self._dualcoef_

    @_dual_coef_.setter
    def _dual_coef_(self, value):
        self._dualcoef_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.dual_coef_ = value
            self._onedal_estimator._onedal_model = None

    @_dual_coef_.deleter
    def _dual_coef_(self):
        del self._dualcoef_

    @property
    def intercept_(self):
        return self._icept_

    @intercept_.setter
    def intercept_(self, value):
        self._icept_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.intercept_ = value
            self._onedal_estimator._onedal_model = None

    @intercept_.deleter
    def intercept_(self):
        del self._icept_

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        patching_status.and_conditions([(False, "GPU offloading is not supported.")])
        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        if method_name == "fit":
            xp, _ = get_namespace(*data)
            _, _, sample_weight = data
            patching_status.and_conditions(
                [
                    (
                        self.kernel in ["linear", "rbf", "poly", "sigmoid"],
                        f'Kernel is "{self.kernel}" while '
                        'only "linear", "rbf", "poly" and "sigmoid" are supported.',
                    ),
                    (
                        sample_weight is None
                        or (
                            xp.all((sw := xp.asarray(sample_weight)) >= 0)
                            and not xp.all(sw == 0)
                        ),
                        "negative or all zero weights are not supported",
                    ),
                ]
            )
            return patching_status
        elif method_name in self._n_jobs_supported_onedal_methods:
            # _n_jobs_supported_onedal_methods is different for classifiers vs regressors
            # and can be easily used to find which methods have underlying direct onedal
            # support.
            patching_status.and_conditions(
                [(hasattr(self, "_onedal_estimator"), "oneDAL model was not trained.")]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")

    def _svm_sample_weight_check(self, sample_weight, y, xp):
        pass

    def _compute_gamma_sigma(self, X):
        # only run extended conversion if kernel is not linear
        # set to a value = 1.0, so gamma will always be passed to
        # the onedal estimator as a float type. This replicates functionality
        # directly out of scikit-learn to enable various variable gamma values.
        if self.kernel == "linear":
            return 1.0

        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                if sp.issparse(X):
                    # var = E[X^2] - E[X]^2
                    X_sc = (X.multiply(X)).mean() - (X.mean()) ** 2
                else:
                    xp, _ = get_namespace(X)
                    X_sc = xp.var(X)
                _gamma = 1.0 / (X.shape[1] * float(X_sc)) if X_sc != 0 else 1.0
            elif self.gamma == "auto":
                _gamma = 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(self.gamma)
                )
        else:
            if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
                if isinstance(self.gamma, Real):
                    if self.gamma <= 0:
                        msg = (
                            f"gamma value must be > 0; {self.gamma!r} is invalid. Use"
                            " a positive number or use 'auto' to set gamma to a"
                            " value of 1 / n_features."
                        )
                        raise ValueError(msg)
                    _gamma = self.gamma
                else:
                    msg = (
                        "The gamma value should be set to 'scale', 'auto' or a"
                        f" positive float value. {self.gamma!r} is not a valid option"
                    )
                    raise ValueError(msg)
            else:
                _gamma = self.gamma
        return _gamma

    def _onedal_predict(self, X, queue=None, xp=None):
        if xp is None:
            xp, _ = get_namespace(X)

        X = validate_data(
            self,
            X,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
            reset=False,
        )

        return self._onedal_estimator.predict(X, queue=queue)


class BaseSVC(BaseSVM):

    def _onedal_cpu_supported(self, method_name, *data):
        patching_status = super()._onedal_cpu_supported(method_name, *data)
        if method_name == "fit" and patching_status.get_status() and data[2] is not None:
            xp, _ = get_namespace(*data)
            _, y, sample_weight = data
            y_array = xp.asarray(y)
            sw_array = xp.asarray(sample_weight).ravel()
            y_nonzero = y_array[xp.greater(sw_array[: y_array.size], 0)]
            patching_status.and_conditions(
                [
                    (
                        (xp.any(y_nonzero != y_nonzero[0])),
                        "Invalid input - all samples with positive weights belong to the same class.",
                    )
                ]
            )
        return patching_status

    # overwrite _validate_targets for array API support
    def _onedal_validate_targets(self, X, y):
        xp, is_array_api_compliant = get_namespace(X, y)

        # _validate_targets equivalent:
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = (
            xp.unique_inverse(y_)
            if is_array_api_compliant
            else xp.unique(y_, return_inverse=True)
        )
        self.class_weight_ = _compute_class_weight(self.class_weight, classes=cls, y=y_)
        if cls.shape[0] < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls
        return xp.asarray(y, dtype=X.dtype)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if not sklearn_check_version("1.2"):
            if self.decision_function_shape not in ("ovr", "ovo", None):
                raise ValueError(
                    f"decision_function_shape must be either 'ovr' or 'ovo', "
                    f"got {self.decision_function_shape}."
                )

        xp, _ = get_namespace(X, y, sample_weight)

        X, y = validate_data(
            self,
            X,
            y,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
        )

        y = self._onedal_validate_targets(X, y)

        if (sw_flag := sample_weight is not None) or self.class_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            # oneDAL only accepts sample_weights, apply class_weight directly

        # due to the nature of how sklearn checks nu in NuSVC (by not checking
        # class_weight-adjusted values, this check must be done here regardless)
        # Logic can be found in libsvm/svm.cpp in scikit-learn.
        self._svm_sample_weight_check(sample_weight, y, xp)

        if self.class_weight is not None:
            if sw_flag:
                # make a copy in order to do modifications of the values
                sample_weight = xp.copy(sample_weight)
            # This for loop is O(n*m) where n is # of classes and m # of samples
            # sklearn's compute_sample_weight (roughly equivalent function) uses
            # np.searchsorted which is roughly O((log(n)*m) but unavailable in
            # the array API standard. Be wary of large class counts.
            for i, v in enumerate(self.class_weight_):
                sample_weight[y == i] *= v

        onedal_params = {
            "C": self.C,
            "nu": self.nu,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._compute_gamma_sigma(X),
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "max_iter": self.max_iter,
            "cache_size": self.cache_size,
        }

        self._onedal_estimator = self._onedal_factory(**onedal_params)
        self._onedal_estimator.fit(
            X, y, sample_weight, class_count=self.classes_.shape[0], queue=queue
        )

        if self.probability:
            self._fit_proba(
                X,
                y,
                sample_weight=sample_weight,
                queue=queue,
            )

        self._save_attributes(X, y, xp=xp)

    def _fit_proba(self, X, y, sample_weight=None, queue=None):
        # TODO: rewrite this method when probabilities output is implemented in oneDAL

        # LibSVM uses the random seed to control cross-validation for probability generation
        # CalibratedClassifierCV with "prefit" does not use an RNG nor a seed. This may
        # impact users without their knowledge, so display a warning.
        if self.random_state is not None:
            warnings.warn(
                "random_state does not influence oneDAL SVM results",
                RuntimeWarning,
            )

        params = self.get_params()
        params["probability"] = False
        params["decision_function_shape"] = "ovr"
        clf_base = self.__class__(**params)

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg["target_offload"] = queue
        with config_context(**cfg):
            clf_base.fit(X, y)

            # Forced use of FrozenEstimator starting in sklearn 1.6
            if sklearn_check_version("1.6"):
                clf_base = FrozenEstimator(clf_base)

                self.clf_prob = CalibratedClassifierCV(
                    clf_base,
                    ensemble=False,
                    method="sigmoid",
                )
                # see custom stopgap solution defined above
                _prefit_CalibratedClassifierCV_fit(
                    self.clf_prob, X, y, sample_weight=sample_weight
                )
            else:

                self.clf_prob = CalibratedClassifierCV(
                    clf_base,
                    ensemble=False,
                    cv="prefit",
                    method="sigmoid",
                ).fit(X, y, sample_weight=sample_weight)

    def _save_attributes(self, X, y, xp=np):
        self.support_vectors_ = self._onedal_estimator.support_vectors_

        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.support_ = xp.asarray(self._onedal_estimator.support_, dtype=xp.int64)

        self._icept_ = self._onedal_estimator.intercept_
        self._sparse = False
        self.fit_status_ = 0
        self.shape_fit_ = X.shape

        self._gamma = self._onedal_estimator.gamma
        length = (self.classes_.shape[0] ** 2 - self.classes_.shape[0]) // 2

        if self.probability:
            # Parameter learned in Platt scaling, exposed as probA_ and probB_
            # via the sklearn SVM estimator
            self._probA = xp.zeros(length)
            self._probB = xp.zeros(length)
        else:
            self._probA = xp.empty(0)
            self._probB = xp.empty(0)

        self._dualcoef_ = self.dual_coef_

        indices = xp.take(y, self.support_, axis=0)
        self._n_support = xp.zeros_like(self.classes_, dtype=xp.int64)
        for i in range(self.classes_.shape[0]):
            self._n_support[i] = xp.sum(
                xp.asarray(indices == i, dtype=xp.int64), dtype=xp.int64
            )

        if sklearn_check_version("1.1"):
            self.n_iter_ = xp.full((length,), self._onedal_estimator.n_iter_)

    def _onedal_predict(self, X, queue=None):
        sv = self.support_vectors_

        xp, _ = get_namespace(X)

        # sklearn conformance >1.0, with array API conversion
        # https://github.com/scikit-learn/scikit-learn/pull/21336
        if (
            not self._sparse
            and sv.shape[0] > 0
            and xp.sum(self._n_support) != sv.shape[0]
        ):
            raise ValueError(
                "The internal representation " f"of {self.__class__.__name__} was altered"
            )

        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when " "decision_function_shape is 'ovo'"
            )

        if (
            self.break_ties
            and self.decision_function_shape == "ovr"
            and self.classes_.shape[0] > 2
        ):
            res = xp.argmax(self._onedal_decision_function(X, queue=queue), axis=1)
        else:
            res = super()._onedal_predict(X, queue=queue, xp=xp)

        # the extensive reshaping here comes from the previous implementation, and
        # should be sorted out, as this is inefficient and likely can be reduced
        res = xp.asarray(res, dtype=xp.int64)
        if self.classes_.shape[0] == 2:
            res = xp.reshape(res, (-1,))

        return xp.reshape(xp.take(xp.asarray(self.classes_), res), (-1,))

    def _onedal_ovr_decision_function(self, decision_function, n_classes, xp=None):
        # This function is legacy from the original implementation and needs
        # to be refactored.

        predictions = xp.asarray(decision_function < 0, dtype=decision_function.dtype)
        confidences = -decision_function

        if xp is None:
            xp, _ = get_namespace(decision_function)
        # use `zeros_like` to support correct device allocation while still
        # supporting numpy < 1.26
        votes = xp.full_like(decision_function[:, :n_classes], n_classes)
        sum_of_confidences = xp.zeros_like(votes)

        # This is extraordinarily bad, as its doing strided access behind
        # two python for loops. Its the main math converting an ovo to ovr.
        k = 0
        for i in range(n_classes):
            votes[:, i] -= i + 1
            for j in range(i + 1, n_classes):
                sum_of_confidences[:, i] -= confidences[:, k]
                sum_of_confidences[:, j] += confidences[:, k]
                votes[:, i] -= predictions[:, k]
                votes[:, j] += predictions[:, k]
                k += 1

        transformed_confidences = sum_of_confidences / (
            3 * (xp.abs(sum_of_confidences) + 1)
        )
        return votes + transformed_confidences

    def _onedal_decision_function(self, X, queue=None):
        xp, _ = get_namespace(X)

        X = validate_data(
            self,
            X,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
            reset=False,
        )

        sv = self.support_vectors_
        if (
            not self._sparse
            and sv.shape[0] > 0
            and xp.sum(self._n_support) != sv.shape[0]
        ):
            raise ValueError(
                "The internal representation " f"of {self.__class__.__name__} was altered"
            )

        decision_function = self._onedal_estimator.decision_function(X, queue=queue)

        lencls = self.classes_.shape[0]
        if lencls == 2:
            decision_function = xp.reshape(decision_function, (-1,))
        elif lencls > 2 and self.decision_function_shape == "ovr":
            decision_function = self._onedal_ovr_decision_function(
                decision_function, lencls, xp
            )

        return decision_function

    def _onedal_predict_proba(self, X, queue=None):
        if not hasattr(self, "clf_prob"):
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg["target_offload"] = queue
        with config_context(**cfg):
            return self.clf_prob.predict_proba(X)

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_BaseSVC.predict,
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
                "sklearn": _sklearn_BaseSVC.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    @wrap_output_data
    def decision_function(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "decision_function",
            {
                "onedal": self.__class__._onedal_decision_function,
                "sklearn": _sklearn_BaseSVC.decision_function,
            },
            X,
        )

    @available_if(_sklearn_BaseSVC._check_proba)
    @wraps(_sklearn_BaseSVC.predict_proba, assigned=["__doc__"])
    def predict_proba(self, X):
        check_is_fitted(self)
        return self._predict_proba(X)

    @available_if(_sklearn_BaseSVC._check_proba)
    @wraps(_sklearn_BaseSVC.predict_log_proba, assigned=["__doc__"])
    def predict_log_proba(self, X):
        xp, _ = get_namespace(X)

        return xp.log(self.predict_proba(X))

    @wrap_output_data
    def _predict_proba(self, X):
        return dispatch(
            self,
            "_predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_BaseSVC.predict_proba,
            },
            X,
        )

    predict.__doc__ = _sklearn_BaseSVC.predict.__doc__
    decision_function.__doc__ = _sklearn_BaseSVC.decision_function.__doc__
    score.__doc__ = _sklearn_BaseSVC.score.__doc__


class BaseSVR(BaseSVM):

    # overwrite _validate_targets for array API support
    def _onedal_validate_targets(self, X, y):
        # this replicates sklearn's `_valdiate_targets` but with X
        # to prevent unnecessary dtype conversions
        xp, is_array_api_compliant = get_namespace(X, y)

        if not is_array_api_compliant:
            return column_or_1d(y, warn=True).astype(X.dtype, copy=False)

        return xp.astype(column_or_1d(y, warn=True), X.dtype, copy=False)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        xp, is_ = get_namespace(X, y, sample_weight)

        X, y = validate_data(
            self,
            X,
            y,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
        )

        y = self._onedal_validate_targets(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        onedal_params = {
            "C": self.C,
            "nu": self.nu,
            "epsilon": self.epsilon,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._compute_gamma_sigma(X),
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
        }

        self._onedal_estimator = self._onedal_factory(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)
        self._save_attributes(X, xp=xp)

    def _save_attributes(self, X, xp=None):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = X.shape
        self.support_ = xp.asarray(self._onedal_estimator.support_, dtype=xp.int64)

        self._icept_ = self._onedal_estimator.intercept_
        self._n_support = xp.asarray([self.support_vectors_.shape[0]], dtype=xp.int64)

        self._sparse = False
        self._gamma = self._onedal_estimator.gamma
        self._probA = None
        self._probB = None

        if sklearn_check_version("1.1"):
            self.n_iter_ = self._onedal_estimator.n_iter_

        self._dualcoef_ = self.dual_coef_

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_BaseLibSVM.predict,
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
                "sklearn": RegressorMixin.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    predict.__doc__ = _sklearn_BaseLibSVM.predict.__doc__
    score.__doc__ = RegressorMixin.score.__doc__
