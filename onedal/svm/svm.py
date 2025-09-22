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
from onedal.utils import _sycl_queue_manager as QM

from ..common._estimator_checks import _check_is_fitted
from ..datatypes import from_table, to_table


class BaseSVM(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        C,
        nu,
        epsilon,
        kernel="rbf",
        *,
        degree,
        gamma,
        coef0,
        tol,
        shrinking,
        cache_size,
        max_iter,
        tau,
        class_weight,
        algorithm,
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
        self.class_weight = class_weight
        self.algorithm = algorithm
        self._onedal_model = None

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def infer(self, *args, **kwargs): ...

    def _get_onedal_params(self, dtype):
        max_iter = 10000 if self.max_iter == -1 else self.max_iter
        # TODO: remove this workaround
        # when oneDAL SVM starts support of 'n_iterations' result
        self.n_iter_ = 1 if max_iter < 1 else max_iter
        class_count = 0 if self.classes_ is None else len(self.classes_)

        return {
            "fptype": dtype,
            "c": self.C,
            "nu": self.nu,
            "epsilon": self.epsilon,
            "kernel": self.kernel,
            "degree": self.degree,
            "shift": self.coef0 if self.kernel != "linear" else 0.0,
            "scale": self.gamma if self.kernel != "linear" else 1.0,
            "sigma": np.sqrt(0.5 / self.gamma) if self.kernel != "linear" else 1.0,
            "accuracy_threshold": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iteration_count": int(max_iter),
            "tau": self.tau,
            "method": self.algorithm,
            "class_count": class_count,
        }

    @supports_queue
    def fit(self, X, y, sample_weight=None, queue=None):

        if sample_weight is not None:
            if self.class_weight_ is not None:
                for i, v in enumerate(self.class_weight_):
                    # needs an array API fix here
                    sample_weight[y == i] *= v
            data = (X, y, sample_weight)
        else:
            data = (X, y)
        self._sparse = sp.issparse(X)

        data_t = to_table(*data, queue=QM.get_global_queue())
        params = self._get_onedal_params(data_t[0].dtype)
        result = self.train(params, *data_t)

        if self._sparse:
            self.dual_coef_ = sp.csr_matrix(from_table(result.coeffs).T)
            self.support_vectors_ = sp.csr_matrix(from_table(result.support_vectors))
        else:
            self.dual_coef_ = from_table(result.coeffs, like=X).T
            self.support_vectors_ = from_table(result.support_vectors, like=X)

        self.intercept_ = from_table(result.biases, like=X)[0, ...]
        self.support_ = from_table(result.support_indices, like=X)[0, ...]

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
            if not sp.isspmatrix(X):
                X = sp.csr_matrix(X)
            else:
                X.sort_indices()
        elif sp.issparse(X) and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__
            )

        X = to_table(X, queue=QM.get_global_queue())
        params = self._get_onedal_params(X)

        if self._onedal_model is None:
            self._onedal_model = self._create_model()

        return self.infer(params, self._onedal_model, X)

    def predict(self, X, queue=None):
        return from_table(self._infer(X, queue).responses, like=X)

    def decision_function(self, X, queue=None):
        return from_table(self._infer(X, queue).decision_function, like=X)


class SVR(BaseSVM):

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):

        # This hard-codes nu=.5, which may be bad. Needs to be investigated
        super().__init__(
            C=C,
            nu=0.5,
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
            class_weight=None,
            algorithm=algorithm,
        )

    @bind_default_backend("svm.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.regression")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.regression")
    def model(self): ...

    def predict(self, X, queue=None):
        # return 1-dimensional output from 2d oneDAL table
        return super().predict(X)[0, ...]


class SVC(BaseSVM):

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        class_weight=None,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=0.5,
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=class_weight,
            algorithm=algorithm,
        )

    def _create_model(self):
        m = super()._create_model()
        m.first_class_response, m.second_class_response = 0, 1
        return m

    @bind_default_backend("svm.classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.classification")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.classification")
    def model(self): ...


class NuSVR(BaseSVM):

    def __init__(
        self,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
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
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=None,
            algorithm=algorithm,
        )

    @bind_default_backend("svm.nu_regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_regression")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_regression")
    def model(self): ...

    def predict(self, X, queue=None):
        # return only a 1-dimensional output from 2d oneDAL table
        return self._predict(X)[0, ...]


class NuSVC(BaseSVM):

    def __init__(
        self,
        nu=0.5,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        class_weight=None,
        algorithm="thunder",
    ):
        super().__init__(
            C=1.0,
            nu=nu,
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=class_weight,
            algorithm=algorithm,
        )

    def _create_model(self):
        m = super()._create_model()
        m.first_class_response, m.second_class_response = 0, 1
        return m

    @bind_default_backend("svm.nu_classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_classification")
    def infer(self, *args, **kwargs): ...

    @bind_default_backend("svm.nu_classification")
    def model(self): ...
