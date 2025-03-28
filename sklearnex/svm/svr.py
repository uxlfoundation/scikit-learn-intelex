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
from sklearn.svm import SVR as _sklearn_SVR
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.svm import SVR as onedal_SVR

from .._device_offload import dispatch, wrap_output_data
from ..utils.validation import validate_data
from ._common import BaseSVR


@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class SVR(BaseSVR, _sklearn_SVR):
    __doc__ = _sklearn_SVR.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_SVR._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif self.C <= 0:
            # else if added to correct issues with
            # sklearn tests:
            # svm/tests/test_sparse.py::test_error
            # svm/tests/test_svm.py::test_bad_input
            # for sklearn versions < 1.2 (i.e. without
            # validate_params parameter checking)
            # Without this, a segmentation fault with
            # Windows fatal exception: access violation
            # occurs
            raise ValueError("C <= 0")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_SVR.fit,
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
                "sklearn": _sklearn_SVR.predict,
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
                "sklearn": _sklearn_SVR.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        X, _, sample_weight = self._onedal_fit_checks(X, y, sample_weight)
        onedal_params = {
            "C": self.C,
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

        self._onedal_estimator = onedal_SVR(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            ensure_all_finite=False,
            accept_sparse="csr",
            reset=False,
        )
        return self._onedal_estimator.predict(X, queue=queue)

    fit.__doc__ = _sklearn_SVR.fit.__doc__
    predict.__doc__ = _sklearn_SVR.predict.__doc__
    score.__doc__ = _sklearn_SVR.score.__doc__
