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
from sklearn.svm import NuSVR as _sklearn_NuSVR
from sklearn.utils.validation import (
    _deprecate_positional_args,
    check_array,
    check_is_fitted,
)

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.svm import NuSVR as onedal_NuSVR

from .._device_offload import dispatch, wrap_output_data
from ..utils.validation import validate_data
from ._common import BaseSVR


@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class NuSVR(BaseSVR, _sklearn_NuSVR):
    __doc__ = _sklearn_NuSVR.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_NuSVR._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        tol=1e-3,
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
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif self.nu <= 0 or self.nu > 1:
            # else if added to correct issues with
            # sklearn tests:
            # svm/tests/test_sparse.py::test_error
            # svm/tests/test_svm.py::test_bad_input
            # for sklearn versions < 1.2 (i.e. without
            # validate_params parameter checking)
            # Without this, a segmentation fault with
            # Windows fatal exception: access violation
            # occurs
            raise ValueError("nu <= 0 or nu > 1")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_NuSVR.fit,
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
                "sklearn": _sklearn_NuSVR.predict,
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
                "sklearn": _sklearn_NuSVR.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        X, _, sample_weight = self._onedal_fit_checks(X, y, sample_weight)
        onedal_params = {
            "C": self.C,
            "nu": self.nu,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._compute_gamma_sigma(X),
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
        }

        self._onedal_estimator = onedal_NuSVR(**onedal_params)
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

    fit.__doc__ = _sklearn_NuSVR.fit.__doc__
    predict.__doc__ = _sklearn_NuSVR.predict.__doc__
    score.__doc__ = _sklearn_NuSVR.score.__doc__
