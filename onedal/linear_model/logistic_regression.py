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

from abc import ABCMeta, abstractmethod

import numpy as np

from onedal import onedal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend

from ..common._estimator_checks import _check_is_fitted
from ..datatypes import from_table, to_table
from ..utils.validation import _check_n_features, _is_csr, _num_features


class BaseLogisticRegression(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, tol, C, fit_intercept, solver, max_iter, algorithm):
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.algorithm = algorithm

    @bind_default_backend("logistic_regression.classification")
    def train(self, params, X, y, queue=None): ...

    @bind_default_backend("logistic_regression.classification")
    def infer(self, params, model, X, queue=None): ...

    @bind_default_backend("logistic_regression.classification")
    def model(self): ...

    def _get_onedal_params(self, is_csr, dtype=np.float32):
        intercept = "intercept|" if self.fit_intercept else ""
        return {
            "fptype": dtype,
            "method": "sparse" if is_csr else self.algorithm,
            "intercept": self.fit_intercept,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "C": self.C,
            "optimizer": self.solver,
            "result_option": (
                intercept
                + "coefficients|iterations_count"
                + ("|inner_iterations_count" if self.solver == "newton-cg" else "")
            ),
        }

    @supports_queue
    def fit(self, X, y, queue=None):

        is_csr = _is_csr(X)

        self.n_features_in_ = _num_features(X, fallback_1d=True)

        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(is_csr, X_table.dtype)

        result = self.train(params, X_table, y_table)

        self._onedal_model = result.model

        self.n_iter_ = np.array([result.iterations_count])

        # _n_inner_iter is the total number of cg-solver iterations
        if onedal_check_version(2024, 3, 0) and self.solver == "newton-cg":
            self._n_inner_iter = result.inner_iterations_count

        coeff = from_table(result.model.packed_coefficients, like=X)
        self.coef_, self.intercept_ = coeff[:, 1:], coeff[:, 0]

        return self

    def _create_model(self, coef_, intercept_, xp):
        model = self.model()
        # Packed coefficients have shape (1, n_features_ + 1), the first element is a placeholder for bias.
        # If fit_intercept is set to False the first element is set to 0.
        if self.fit_intercept:
            intercept_ = xp.reshape(intercept_, (-1, 1))
            if xp is np:
                # workaround for numpy<2
                packed_coefficients = xp.concatenate([intercept_, coef_], axis=1)
            else:
                packed_coefficients = xp.concat([intercept_, coef_], axis=1)
        else:
            if hasattr(coef_, "device"):
                packed_coefficients = xp.zeros(
                    (coef_.shape[0], coef_.shape[1] + 1), device=coef_.device
                )
            else:
                # Note: on numpy<2, these objects and functions do not have
                # the 'device' attribute/argument.
                packed_coefficients = xp.zeros((coef_.shape[0], coef_.shape[1] + 1))
            packed_coefficients[:, 1:] = coef_

        model.packed_coefficients = to_table(packed_coefficients)
        self.coef_ = coef_
        self.intercept_ = intercept_
        self._onedal_model = model

    def _infer(self, X, queue=None):
        _check_is_fitted(self)

        is_csr = _is_csr(X)

        _check_n_features(self, X, False)

        assert hasattr(self, "_onedal_model")

        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(is_csr, X_table.dtype)

        result = self.infer(params, self._onedal_model, X_table)
        return result

    @supports_queue
    def predict(self, X, queue=None, classes=None):
        result = self._infer(X, queue)

        # Starting from sklearn 1.9 type of predicted labels should match the type of self.classes_
        # In general case, classes attribute is provided from sklearnex estimator
        # In case it's not provided, result would be of the same type as X
        y = from_table(result.responses, like=classes if classes is not None else X)
        return y

    @supports_queue
    def predict_proba(self, X, queue=None):
        result = self._infer(X, queue)
        y = from_table(result.probabilities, like=X)
        return y


class LogisticRegression(BaseLogisticRegression):

    def __init__(
        self,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        solver="newton-cg",
        max_iter=100,
        *,
        algorithm="dense_batch",
        **kwargs,
    ):
        super().__init__(
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            algorithm=algorithm,
        )
