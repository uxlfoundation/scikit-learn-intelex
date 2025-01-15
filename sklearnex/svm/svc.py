# ==============================================================================
# Copyright 2021 Intel Corporation
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

import numpy as np
from scipy import sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as _sklearn_SVC
from sklearn.utils.validation import (
    _deprecate_positional_args,
    check_array,
    check_is_fitted,
)

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain
from ..utils._array_api import get_namespace
from ._common import BaseSVC

if sklearn_check_version("1.0"):
    from sklearn.utils.metaestimators import available_if

from onedal.svm import SVC as onedal_SVC

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = BaseSVC._validate_data


@control_n_jobs(
    decorated_methods=["fit", "predict", "_predict_proba", "decision_function", "score"]
)
class SVC(_sklearn_SVC, BaseSVC):
    __doc__ = _sklearn_SVC.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_SVC._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_SVC.fit,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_SVC.predict,
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
                "sklearn": _sklearn_SVC.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    if sklearn_check_version("1.0"):

        @available_if(_sklearn_SVC._check_proba)
        def predict_proba(self, X):
            """
            Compute probabilities of possible outcomes for samples in X.

            The model need to have probability information computed at training
            time: fit with attribute `probability` set to True.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                For kernel="precomputed", the expected shape of X is
                (n_samples_test, n_samples_train).

            Returns
            -------
            T : ndarray of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in
                the model. The columns correspond to the classes in sorted
                order, as they appear in the attribute :term:`classes_`.

            Notes
            -----
            The probability model is created using cross validation, so
            the results can be slightly different than those obtained by
            predict. Also, it will produce meaningless results on very small
            datasets.
            """
            check_is_fitted(self)
            return self._predict_proba(X)

        @available_if(_sklearn_SVC._check_proba)
        def predict_log_proba(self, X):
            """Compute log probabilities of possible outcomes for samples in X.

            The model need to have probability information computed at training
            time: fit with attribute `probability` set to True.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features) or \
                    (n_samples_test, n_samples_train)
                For kernel="precomputed", the expected shape of X is
                (n_samples_test, n_samples_train).

            Returns
            -------
            T : ndarray of shape (n_samples, n_classes)
                Returns the log-probabilities of the sample for each class in
                the model. The columns correspond to the classes in sorted
                order, as they appear in the attribute :term:`classes_`.

            Notes
            -----
            The probability model is created using cross validation, so
            the results can be slightly different than those obtained by
            predict. Also, it will produce meaningless results on very small
            datasets.
            """
            xp, _ = get_namespace(X)

            return xp.log(self.predict_proba(X))

    else:

        @property
        def predict_proba(self):
            self._check_proba()
            check_is_fitted(self)
            return self._predict_proba

        def _predict_log_proba(self, X):
            xp, _ = get_namespace(X)
            return xp.log(self.predict_proba(X))

        predict_proba.__doc__ = _sklearn_SVC.predict_proba.__doc__

    @wrap_output_data
    def _predict_proba(self, X):
        sklearn_pred_proba = (
            _sklearn_SVC.predict_proba
            if sklearn_check_version("1.0")
            else _sklearn_SVC._predict_proba
        )

        return dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": sklearn_pred_proba,
            },
            X,
        )

    @wrap_output_data
    def decision_function(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "decision_function",
            {
                "onedal": self.__class__._onedal_decision_function,
                "sklearn": _sklearn_SVC.decision_function,
            },
            X,
        )

    decision_function.__doc__ = _sklearn_SVC.decision_function.__doc__

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        if len(data) > 1:
            self._class_count = len(np.unique(data[1]))
        self._is_sparse = sp.issparse(data[0])
        conditions = [
            (
                self.kernel in ["linear", "rbf"],
                f'Kernel is "{self.kernel}" while '
                '"linear" and "rbf" are only supported on GPU.',
            ),
            (self.class_weight is None, "Class weight is not supported on GPU."),
            (not self._is_sparse, "Sparse input is not supported on GPU."),
            (self._class_count == 2, "Multiclassification is not supported on GPU."),
        ]
        if method_name == "fit":
            patching_status.and_conditions(conditions)
            return patching_status
        if method_name in ["predict", "predict_proba", "decision_function", "score"]:
            conditions.append(
                (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained")
            )
            patching_status.and_conditions(conditions)
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")

    def _get_sample_weight(self, X, y, sample_weight=None):
        sample_weight = super()._get_sample_weight(X, y, sample_weight)
        if sample_weight is None:
            return sample_weight

        if np.any(sample_weight <= 0) and len(np.unique(y[sample_weight > 0])) != len(
            self.classes_
        ):
            raise ValueError(
                "Invalid input - all samples with positive weights "
                "belong to the same class"
                if sklearn_check_version("1.2")
                else "Invalid input - all samples with positive weights "
                "have the same label."
            )
        return sample_weight

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        X, _, weights = self._onedal_fit_checks(X, y, sample_weight)
        onedal_params = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._compute_gamma_sigma(X),
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "break_ties": self.break_ties,
            "decision_function_shape": self.decision_function_shape,
        }

        self._onedal_estimator = onedal_SVC(**onedal_params)
        self._onedal_estimator.fit(X, y, weights, queue=queue)

        if self.probability:
            self._fit_proba(
                X,
                y,
                sample_weight=sample_weight,
                queue=queue,
            )

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        if sklearn_check_version("1.0"):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=False,
                ensure_2d=False,
                accept_sparse="csr",
                reset=False,
            )
        else:
            X = check_array(
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=False,
                accept_sparse="csr",
            )
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_predict_proba(self, X, queue=None):
        if getattr(self, "clf_prob", None) is None:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
        from .._config import config_context, get_config

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg["target_offload"] = queue
        with config_context(**cfg):
            return self.clf_prob.predict_proba(X)

    def _onedal_decision_function(self, X, queue=None):
        if sklearn_check_version("1.0"):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=False,
                accept_sparse="csr",
                reset=False,
            )
        else:
            X = check_array(
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=False,
                accept_sparse="csr",
            )
        return self._onedal_estimator.decision_function(X, queue=queue)

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    fit.__doc__ = _sklearn_SVC.fit.__doc__
    predict.__doc__ = _sklearn_SVC.predict.__doc__
    decision_function.__doc__ = _sklearn_SVC.decision_function.__doc__
    score.__doc__ = _sklearn_SVC.score.__doc__
