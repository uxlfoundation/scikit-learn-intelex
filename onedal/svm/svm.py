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

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend

from ..common._estimator_checks import _check_is_fitted
from ..datatypes import from_table, to_table
from ..utils.validation import _is_csr


class BaseSVM(metaclass=ABCMeta):

    def __init__(
        self,
        C,  # Depending on the child class, C, nu, and/or epsilon are not used
        nu,
        epsilon,
        kernel="rbf",
        *,
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.tol = tol
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.tau = tau
        self.algorithm = algorithm
        self._onedal_model = None

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def infer(self, *args, **kwargs): ...

    def _get_onedal_params(self, X):
        max_iter = 10000 if self.max_iter == -1 else self.max_iter
        # TODO: remove this workaround
        # when oneDAL SVM starts support of 'n_iterations' result
        self.n_iter_ = max(1, max_iter)
        # if gamma is not given as a value, use sklearn's "auto"
        gamma = 1 / X.shape[1] if self.gamma is None else self.gamma
        return {
            "fptype": X.dtype,
            "c": self.C,
            "nu": self.nu,
            "epsilon": self.epsilon,
            "kernel": self.kernel,
            "degree": self.degree,
            "shift": self.coef0 if self.kernel != "linear" else 0.0,
            "scale": gamma if self.kernel != "linear" else 1.0,
            "sigma": np.sqrt(0.5 / gamma) if self.kernel != "linear" else 1.0,
            "accuracy_threshold": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iteration_count": int(max_iter),
            "tau": self.tau,
            "method": self.algorithm,
            "class_count": self.class_count_,
        }

    @supports_queue
    def fit(self, X, y, sample_weight=None, class_count=0, queue=None):
        # oneDAL expects that the user has a priori knowledge of the y data
        # and has placed them in a oneDAL-acceptable format, most important
        # for this is the number of classes.
        self.class_count_ = class_count

        data = (X, y) if sample_weight is None else (X, y, sample_weight)

        self._sparse = sp.issparse(X)

        data_t = to_table(*data, queue=queue)
        params = self._get_onedal_params(data_t[0])
        result = self.train(params, *data_t)

        if self._sparse:
            self.dual_coef_ = sp.csr_matrix(from_table(result.coeffs).T)
            self.support_vectors_ = sp.csr_matrix(from_table(result.support_vectors))
        else:
            self.dual_coef_ = from_table(result.coeffs, like=X).T
            self.support_vectors_ = from_table(result.support_vectors, like=X)

        self.intercept_ = from_table(result.biases, like=X)[0, ...]
        self.support_ = from_table(result.support_indices, like=X)[:, 0]

        self._onedal_model = result.model
        return self

    def _create_model(self):
        m = self.model()

        m.support_vectors = to_table(self.support_vectors_)
        m.coeffs = to_table(self.dual_coef_.T)
        m.biases = to_table(self.intercept_)
        return m

    @supports_queue
    def _infer(self, X, queue=None):
        _check_is_fitted(self)

        if self._sparse:
            if not _is_csr(X):
                X = sp.csr_array(X) if hasattr(sp, "csr_array") else sp.csr_matrix(X)
            else:
                X.sort_indices()

        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X)

        if self._onedal_model is None:
            self._onedal_model = self._create_model()

        return self.infer(params, self._onedal_model, X)

    def predict(self, X, queue=None):
        return from_table(self._infer(X, queue=queue).responses, like=X)[:, 0]


class SVR(BaseSVM):

    def __init__(
        self,
        C=1.0,
        nu=None,  # not used
        epsilon=0.1,
        kernel="rbf",
        *,
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):

        super().__init__(
            C=C,
            nu=nu,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            algorithm=algorithm,
        )

    @bind_default_backend("svm.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.regression")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.regression")
    def model(self): ...

    def _get_onedal_params(self, X):
        params = super()._get_onedal_params(X)
        # The nu parameter is not set
        params.pop("nu")
        return params


class SVC(BaseSVM):

    def __init__(
        self,
        C=1.0,
        nu=None,  # not used
        epsilon=None,  # not used
        kernel="rbf",
        *,
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=nu,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            algorithm=algorithm,
        )

    def _create_model(self):
        m = super()._create_model()
        m.first_class_response, m.second_class_response = 0, 1
        return m

    def _get_onedal_params(self, X):
        params = super()._get_onedal_params(X)
        # The nu and epsilon parameter are not used
        params.pop("nu")
        params.pop("epsilon")
        return params

    @bind_default_backend("svm.classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.classification")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.classification")
    def model(self): ...

    def decision_function(self, X, queue=None):
        return from_table(self._infer(X, queue=queue).decision_function, like=X)


class NuSVR(BaseSVM):

    def __init__(
        self,
        nu=0.5,
        C=1.0,
        epsilon=None,  # not used
        kernel="rbf",
        *,
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=nu,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            algorithm=algorithm,
        )

    def _get_onedal_params(self, X):
        params = super()._get_onedal_params(X)
        # The epsilon parameter is not used
        params.pop("epsilon")
        return params

    @bind_default_backend("svm.nu_regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_regression")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_regression")
    def model(self): ...


class NuSVC(BaseSVM):

    def __init__(
        self,
        C=None,  # not used
        nu=0.5,
        epsilon=None,  # not used
        kernel="rbf",
        *,
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=nu,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            algorithm=algorithm,
        )

    def _create_model(self):
        m = super()._create_model()
        m.first_class_response, m.second_class_response = 0, 1
        return m

    def _get_onedal_params(self, X):
        params = super()._get_onedal_params(X)
        # The C and epsilon parameters are not used
        params.pop("c")
        params.pop("epsilon")
        return params

    @bind_default_backend("svm.nu_classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_classification")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_classification")
    def model(self): ...

    def decision_function(self, X, queue=None):
        return from_table(self._infer(X, queue=queue).decision_function, like=X)
