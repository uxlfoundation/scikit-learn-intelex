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
from numbers import Number

import numpy as np

from daal4py.sklearn._utils import daal_check_version, get_dtype, make2d
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from ..common._estimator_checks import _check_is_fitted
from ..datatypes import from_table, to_table
from ..utils.validation import (
    _check_n_features,
    _is_csr,
    _num_features
)


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
        
        # TODO: we might need to add array api support here later
        self.n_iter_ = np.array([result.iterations_count])

        # _n_inner_iter is the total number of cg-solver iterations
        if daal_check_version((2024, "P", 300)) and self.solver == "newton-cg":
            self._n_inner_iter = result.inner_iterations_count

        coeff = from_table(result.model.packed_coefficients, like=X)
        self.coef_, self.intercept_ = coeff[:, 1:], coeff[:, 0]

        return self

    # TODO Change logic of _create_model to support behaviour when model
    # was fitted in sklearn and then predict is called with torch.tensor on gpu
    # Same method exists in LinReg
    def _create_model(self):

        m = self.model()

        coefficients = self.coef_
        dtype = get_dtype(coefficients)
        coefficients = np.asarray(coefficients, dtype=dtype)

        if coefficients.ndim == 2:
            n_features_in = coefficients.shape[1]
            assert coefficients.shape[0] == 1
        else:
            n_features_in = coefficients.size

        intercept = self.intercept_
        if not isinstance(intercept, Number):
            intercept = np.asarray(intercept, dtype=dtype)
            assert intercept.size == 1

        coefficients, intercept = make2d(coefficients), make2d(intercept)

        assert coefficients.shape == (1, n_features_in)
        assert intercept.shape == (1, 1)

        desired_shape = (1, n_features_in + 1)
        packed_coefficients = np.zeros(desired_shape, dtype=dtype)

        packed_coefficients[:, 1:] = coefficients
        if self.fit_intercept:
            packed_coefficients[:, 0][:, np.newaxis] = intercept

        m.packed_coefficients = to_table(packed_coefficients, queue=QM.get_global_queue())

        self._onedal_model = m

        return m

    def _infer(self, X, queue=None):
        _check_is_fitted(self)

        is_csr = _is_csr(X)

        _check_n_features(self, X, False)

        if not hasattr(self, "_onedal_model"):
            self._onedal_model = self._create_model()

        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(is_csr, X_table.dtype)

        result = self.infer(params, self._onedal_model, X_table)
        return result

    @supports_queue
    def predict(self, X, queue=None):
        result = self._infer(X, queue)
        y = from_table(result.responses, like=X)
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
