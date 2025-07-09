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

from .._config import _get_config
from ..common._estimator_checks import _check_is_fitted
from ..common._mixin import ClassifierMixin
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import (
    _check_array,
    _check_n_features,
    _check_X_y,
    _is_csr,
    _num_features,
    _type_of_target,
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

    @abstractmethod
    def train(self, params, X, y): ...

    @abstractmethod
    def infer(self, params, X): ...

    # direct access to the backend model constructor
    @abstractmethod
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

    def _fit(self, X, y):
        use_raw_input = _get_config()["use_raw_input"] is True

        sparsity_enabled = daal_check_version((2024, "P", 700))
        if not use_raw_input:
            X, y = _check_X_y(
                X,
                y,
                accept_sparse=sparsity_enabled,
                force_all_finite=True,
                accept_2d_y=False,
                dtype=[np.float64, np.float32],
            )
            if _type_of_target(y) != "binary":
                raise ValueError("Only binary classification is supported")

            self.classes_, y = np.unique(y, return_inverse=True)
            y = y.astype(dtype=np.int32)
        else:
            _, xp, _ = _get_sycl_namespace(X)
            # try catch needed for raw_inputs + array_api data where unlike
            # numpy the way to yield unique values is via `unique_values`
            # This should be removed when refactored for gpu zero-copy
            try:
                self.classes_ = xp.unique(y)
            except AttributeError:
                self.classes_ = xp.unique_values(y)

            n_classes = len(self.classes_)
            if n_classes != 2:
                raise ValueError("Only binary classification is supported")
        is_csr = _is_csr(X)

        self.n_features_in_ = _num_features(X, fallback_1d=True)
        X_table, y_table = to_table(X, y, queue=QM.get_global_queue())
        params = self._get_onedal_params(is_csr, X_table.dtype)

        result = self.train(params, X_table, y_table)

        self._onedal_model = result.model
        self.n_iter_ = np.array([result.iterations_count])

        # _n_inner_iter is the total number of cg-solver iterations
        if daal_check_version((2024, "P", 300)) and self.solver == "newton-cg":
            self._n_inner_iter = result.inner_iterations_count

        coeff = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = coeff[:, 1:], coeff[:, 0]

        return self

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

        intercept = _check_array(
            intercept,
            dtype=[np.float64, np.float32],
            force_all_finite=True,
            ensure_2d=False,
        )
        coefficients = _check_array(
            coefficients,
            dtype=[np.float64, np.float32],
            force_all_finite=True,
            ensure_2d=False,
        )

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

    def _infer(self, X):
        _check_is_fitted(self)

        sparsity_enabled = daal_check_version((2024, "P", 700))

        if not _get_config()["use_raw_input"]:
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                accept_sparse=sparsity_enabled,
                force_all_finite=True,
                ensure_2d=False,
                accept_large_sparse=sparsity_enabled,
            )
        is_csr = _is_csr(X)
        _check_n_features(self, X, False)

        X = make2d(X)

        if hasattr(self, "_onedal_model"):
            model = self._onedal_model
        else:
            model = self._create_model()

        X_table = to_table(X, queue=QM.get_global_queue())
        params = self._get_onedal_params(is_csr, X.dtype)

        result = self.infer(params, model, X_table)
        return result

    def _predict(self, X):
        result = self._infer(X)
        sua_iface, xp, _ = _get_sycl_namespace(X)
        y = from_table(
            result.responses,
            sua_iface=sua_iface,
            sycl_queue=QM.get_global_queue(),
            xp=xp,
        )
        y = xp.take(xp.asarray(self.classes_), xp.reshape(y, (-1,)), axis=0)
        return y

    def _predict_proba(self, X):
        result = result = self._infer(X)
        sua_iface, xp, _ = _get_sycl_namespace(X)
        queue = QM.get_global_queue()
        y = from_table(result.probabilities, sua_iface=sua_iface, sycl_queue=queue, xp=xp)
        return xp.stack([1 - y, y], axis=1)

    def _predict_log_proba(self, X):
        _, xp, _ = _get_sycl_namespace(X)
        y_proba = self._predict_proba(X)
        return xp.log(y_proba)


class LogisticRegression(ClassifierMixin, BaseLogisticRegression):

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

    @bind_default_backend("logistic_regression.classification")
    def train(self, params, X, y, queue=None): ...

    @bind_default_backend("logistic_regression.classification")
    def infer(self, params, X, model, queue=None): ...

    @bind_default_backend("logistic_regression.classification")
    def model(self): ...

    @supports_queue
    def fit(self, X, y, queue=None):
        return self._fit(X, y)

    @supports_queue
    def predict(self, X, queue=None):
        return self._predict(X)

    @supports_queue
    def predict_proba(self, X, queue=None):
        return self._predict_proba(X)

    @supports_queue
    def predict_log_proba(self, X, queue=None):
        return self._predict_log_proba(X)
