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

import numpy as np
from scipy import sparse as sp
from sklearn.svm import SVC as _sklearn_SVC
from sklearn.svm import SVR as _sklearn_SVR
from sklearn.svm import NuSVC as _sklearn_NuSVC
from sklearn.svm import NuSVR as _sklearn_NuSVR
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _deprecate_positional_args

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.svm import SVC as onedal_SVC
from onedal.svm import SVR as onedal_SVR
from onedal.svm import NuSVC as onedal_NuSVC
from onedal.svm import NuSVR as onedal_NuSVR

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..utils._array_api import enable_array_api, get_namespace
from ._base import BaseSVC, BaseSVR

# array API support limited to sklearn 1.5 for regressors due to an incorrect array API
# implementation of `accuracy_score`. Classifiers limited by `_compute_class_weight` to
# sklearn 1.6.


@enable_array_api("1.6")
@control_n_jobs(
    decorated_methods=["fit", "predict", "_predict_proba", "decision_function", "score"]
)
class SVC(BaseSVC, _sklearn_SVC):
    __doc__ = _sklearn_SVC.__doc__
    _onedal_factory = onedal_SVC

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
            raise ValueError("'C' must be strictly positive.")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_SVC.fit,
            },
            X,
            y,
            sample_weight,
        )

        return self

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        conditions = [
            (
                self.kernel in ["linear", "rbf"],
                f'Kernel is "{self.kernel}" while '
                'only "linear" and "rbf" are supported on GPU.',
            ),
            (not sp.issparse(data[0]), "Sparse input is not supported on GPU."),
            (self.class_weight is None, "Class weight is not supported on GPU."),
            (
                len(data) < 2 or type_of_target(data[1]) == "binary",
                "Multiclassification is not supported on GPU.",
            ),
        ]
        if method_name == "fit":
            xp, _ = get_namespace(*data)
            _, _, sample_weight = data
            conditions.append(
                (
                    sample_weight is None
                    or (
                        xp.all((sw := xp.asarray(sample_weight)) >= 0)
                        and not xp.all(sw == 0)
                    ),
                    "negative or all zero weights are not supported",
                )
            )
        elif method_name in self._n_jobs_supported_onedal_methods:
            conditions.append(
                (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained")
            )
        else:
            raise RuntimeError(f"Unknown method {method_name} in {class_name}")

        patching_status.and_conditions(conditions)
        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        conditions = []
        if method_name == "fit":
            xp, _ = get_namespace(*data)
            _, y, sample_weight = data
            y_nonzero = y[sample_weight > xp.full_like(sample_weight, 0)] if sample_weight is not None else y
            conditions.append(
                (
                    (sample_weight is None or not xp.all(y_nonzero == y_nonzero[0])),
                    "Invalid input - all samples with positive weights belong to the same class.",
                )
            )
        patching_status.and_conditions(conditions)
        return patching_status

    fit.__doc__ = _sklearn_SVC.fit.__doc__


@enable_array_api("1.6")
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
            raise ValueError("'nu' must be in the range (0, 1].")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_NuSVC.fit,
            },
            X,
            y,
            sample_weight,
        )

        return self

    def _svm_sample_weight_check(self, sample_weight, y, xp):
        # This provides SVM-specific sample_weight conformance checks
        # It must still check when sample_weight is None to conform to sklearn
        # checks.
        if sample_weight is None:
            if hasattr(xp, "unique_counts"):
                weight_per_class = xp.astype(xp.unique_counts(y)[1], xp.float64)
            else:
                weight_per_class = xp.unique(y, return_counts=True)[1].astype(xp.float64)
            wlcls = weight_per_class.shape[0]
        else:

            weight_per_class = [
                xp.sum(sample_weight[y == class_label])
                for class_label in xp.arange(xp.max(y) + 1, dtype=y.dtype)
            ]
            wlcls = len(weight_per_class)

        for i in range(wlcls):
            for j in range(i + 1, wlcls):
                if self.nu * (weight_per_class[i] + weight_per_class[j]) / 2 > min(
                    weight_per_class[i], weight_per_class[j]
                ):
                    raise ValueError("specified nu is infeasible")

    fit.__doc__ = _sklearn_NuSVC.fit.__doc__


@enable_array_api("1.5")
@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class SVR(BaseSVR, _sklearn_SVR):
    __doc__ = _sklearn_SVR.__doc__
    _onedal_factory = onedal_SVR

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
            raise ValueError("'C' must be strictly positive.")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_SVR.fit,
            },
            X,
            y,
            sample_weight,
        )

        return self

    fit.__doc__ = _sklearn_SVR.fit.__doc__


@enable_array_api("1.5")
@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class NuSVR(BaseSVR, _sklearn_NuSVR):
    __doc__ = _sklearn_NuSVR.__doc__
    _onedal_factory = onedal_NuSVR

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
            raise ValueError("'nu' must be in the range (0, 1].")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_NuSVR.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    fit.__doc__ = _sklearn_NuSVR.fit.__doc__
