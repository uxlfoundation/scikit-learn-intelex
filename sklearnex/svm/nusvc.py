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
from sklearn.svm import NuSVC as _sklearn_NuSVC
from sklearn.utils.validation import _deprecate_positional_args

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.svm import NuSVC as onedal_NuSVC

from .._device_offload import dispatch
from ._common import BaseSVC


@control_n_jobs(
    decorated_methods=["fit", "predict", "_predict_proba", "decision_function", "score"]
)
class NuSVC(BaseSVC, _sklearn_NuSVC):
    __doc__ = _sklearn_NuSVC.__doc__
    _onedal_factory = onedal_NuSVC

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_NuSVC._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        nu=0.5,
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
            nu=nu,
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
                "sklearn": _sklearn_NuSVC.fit,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

        return self

    def _onedal_fit_checks(self, X, y, sample_weight=None):
        X, y, sample_weight = super()._onedal_fit_checks(X, y, sample_weight)
        if sample_weight is not None:
            # This requires adaptation for array API
            weight_per_class = [
                np.sum(sample_weight[y == class_label]) for class_label in np.unique(y)
            ]

            for i in range(len(weight_per_class)):
                for j in range(i + 1, len(weight_per_class)):
                    if self.nu * (weight_per_class[i] + weight_per_class[j]) / 2 > min(
                        weight_per_class[i], weight_per_class[j]
                    ):
                        raise ValueError("specified nu is infeasible")

        return X, y, sample_weight

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        X, _, weights = self._onedal_fit_checks(X, y, sample_weight)

        onedal_params = {
            "nu": self.nu,
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

        self._onedal_estimator = self._onedal_factory(**onedal_params)
        self._onedal_estimator.fit(X, y, weights, queue=queue)

        if self.probability:
            self._fit_proba(
                X,
                y,
                sample_weight=sample_weight,
                queue=queue,
            )

        self._save_attributes()

    fit.__doc__ = _sklearn_NuSVC.fit.__doc__
