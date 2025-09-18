# ==============================================================================
# Copyright 2020 Intel Corporation
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

import numbers

import numpy as np
from scipy import sparse as sp
from sklearn.linear_model._coordinate_descent import ElasticNet as ElasticNet_original
from sklearn.linear_model._coordinate_descent import Lasso as Lasso_original
from sklearn.utils import check_array, check_X_y

import daal4py
from daal4py.sklearn._utils import (
    PatchingConditionsChain,
    get_patch_message,
    getFPType,
    make2d,
    sklearn_check_version,
)

from .._n_jobs_support import control_n_jobs
from ..utils.validation import check_feature_names

if not sklearn_check_version("1.2"):
    from sklearn.linear_model._base import _deprecate_normalize
if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar

import logging

# only for compliance with Sklearn
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize


def _daal4py_check(self, X, y, check_input):
    _fptype = getFPType(X)

    # check alpha
    if self.alpha == 0:
        warnings.warn(
            "With alpha=0, this algorithm does not converge "
            "well. You are advised to use the LinearRegression "
            "estimator",
            stacklevel=2,
        )

    # check l1_ratio
    if (
        not isinstance(self.l1_ratio, numbers.Number)
        or self.l1_ratio < 0
        or self.l1_ratio > 1
    ):
        raise ValueError(
            "l1_ratio must be between 0 and 1; " f"got l1_ratio={self.l1_ratio}"
        )

    # check precompute
    if isinstance(self.precompute, np.ndarray):
        if check_input:
            check_array(self.precompute, dtype=_fptype)
        self.precompute = make2d(self.precompute)
    else:
        if self.precompute not in [False, True, "auto"]:
            raise ValueError(
                "precompute should be one of True, False, "
                "'auto' or array-like. Got %r" % self.precompute
            )

    # check selection
    if self.selection not in ["random", "cyclic"]:
        raise ValueError("selection should be either random or cyclic.")


def _daal4py_fit_enet(self, X, y_, check_input):
    # appropriate checks
    _daal4py_check(self, X, y_, check_input)
    X = make2d(X)
    y = make2d(y_)
    _fptype = getFPType(X)

    # only for dual_gap computation, it is not required for oneAPI
    # Data Analytics Library
    self._X = X
    self.n_features_in_ = X.shape[1]
    self._y = y

    penalty_L1 = np.asarray(self.alpha * self.l1_ratio, dtype=X.dtype)
    penalty_L2 = np.asarray(self.alpha * (1.0 - self.l1_ratio), dtype=X.dtype)
    if penalty_L1.size != 1 or penalty_L2.size != 1:
        raise ValueError("alpha or l1_ratio length is wrong")
    penalty_L1 = penalty_L1.reshape((1, -1))
    penalty_L2 = penalty_L2.reshape((1, -1))

    # normalizing and centering
    X_offset = np.zeros(X.shape[1], dtype=X.dtype)
    X_scale = np.ones(X.shape[1], dtype=X.dtype)
    if y.ndim == 1:
        y_offset = X.dtype.type(0)
    else:
        y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    if sklearn_check_version("1.2"):
        _normalize = False
    else:
        _normalize = self._normalize
    if self.fit_intercept:
        X_offset = np.average(X, axis=0)
        if _normalize:
            if self.copy_X:
                X = np.copy(X) - X_offset
            else:
                X -= X_offset
            X, X_scale = normalize(X, axis=0, copy=False, return_norm=True)
            y_offset = np.average(y, axis=0)
            y = y - y_offset

    # only for compliance with Sklearn
    if (
        isinstance(self.precompute, np.ndarray)
        and self.fit_intercept
        and (
            not np.allclose(X_offset, np.zeros(X.shape[1]))
            or _normalize
            and not np.allclose(X_scale, np.ones(X.shape[1]))
        )
    ):
        if sklearn_check_version("1.4"):
            warnings.warn(
                "Gram matrix was provided but X was centered"
                " to fit intercept: recomputing Gram matrix.",
                UserWarning,
            )
        else:
            warnings.warn(
                "Gram matrix was provided but X was centered"
                " to fit intercept, "
                "or X was normalized : recomputing Gram matrix.",
                UserWarning,
            )

    mse_alg = daal4py.optimization_solver_mse(
        numberOfTerms=X.shape[0], fptype=_fptype, method="defaultDense"
    )
    mse_alg.setup(X, y, None)

    cd_solver = daal4py.optimization_solver_coordinate_descent(
        function=mse_alg,
        fptype=_fptype,
        method="defaultDense",
        selection=self.selection,
        seed=0 if self.random_state is None else self.random_state,
        nIterations=self.max_iter,
        positive=self.positive,
        accuracyThreshold=self.tol,
    )

    # set warm_start
    if self.warm_start and hasattr(self, "coef_") and isinstance(self.coef_, np.ndarray):
        n_rows = y.shape[1]
        n_cols = X.shape[1] + 1
        inputArgument = np.zeros((n_rows, n_cols), dtype=_fptype)
        for i in range(n_rows):
            inputArgument[i][0] = self.intercept_ if (n_rows == 1) else self.intercept_[i]
            inputArgument[i][1:] = (
                self.coef_[:].copy(order="C")
                if (n_rows == 1)
                else self.coef_[i, :].copy(order="C")
            )
        cd_solver.setup(inputArgument)
    doUse_condition = self.copy_X is False or (
        self.fit_intercept and _normalize and self.copy_X
    )
    elastic_net_alg = daal4py.elastic_net_training(
        fptype=_fptype,
        method="defaultDense",
        interceptFlag=(self.fit_intercept is True),
        dataUseInComputation="doUse" if doUse_condition else "doNotUse",
        penaltyL1=penalty_L1,
        penaltyL2=penalty_L2,
        optimizationSolver=cd_solver,
    )
    try:
        if isinstance(self.precompute, np.ndarray):
            elastic_net_res = elastic_net_alg.compute(
                data=X, dependentVariables=y, gramMatrix=self.precompute
            )
        else:
            elastic_net_res = elastic_net_alg.compute(data=X, dependentVariables=y)
    except RuntimeError:
        return None

    # set coef_ and intersept_ results
    elastic_net_model = elastic_net_res.model
    self.daal_model_ = elastic_net_model

    # update coefficients if normalizing and centering
    if self.fit_intercept and _normalize:
        elastic_net_model.Beta[:, 1:] = elastic_net_model.Beta[:, 1:] / X_scale
        elastic_net_model.Beta[:, 0] = (
            y_offset - np.dot(X_offset, elastic_net_model.Beta[:, 1:].T)
        ).T

    coefs = elastic_net_model.Beta

    self.intercept_ = coefs[:, 0].copy(order="C")
    self.coef_ = coefs[:, 1:].copy(order="C")

    # only for compliance with Sklearn
    if y.shape[1] == 1:
        self.coef_ = np.ravel(self.coef_)
    self.intercept_ = np.ravel(self.intercept_)
    if self.intercept_.shape[0] == 1:
        self.intercept_ = self.intercept_[0]

    # set n_iter_
    n_iter = cd_solver.__get_result__().nIterations[0][0]
    if y.shape[1] == 1:
        self.n_iter_ = n_iter
    else:
        self.n_iter_ = np.full(y.shape[1], n_iter)

    # only for compliance with Sklearn
    if self.max_iter == n_iter + 1:
        warnings.warn(
            "Objective did not converge. You might want to "
            "increase the number of iterations.",
            ConvergenceWarning,
        )

    return self


def _daal4py_predict_enet(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)

    elastic_net_palg = daal4py.elastic_net_prediction(
        fptype=_fptype, method="defaultDense"
    )
    if self.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, "
            f"but ElasticNet is expecting "
            f"{self.n_features_in_} features as input"
        )
    elastic_net_res = elastic_net_palg.compute(X, self.daal_model_)

    res = elastic_net_res.prediction

    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)
    return res


def _daal4py_fit_lasso(self, X, y_, check_input):
    # appropriate checks
    _daal4py_check(self, X, y_, check_input)
    X = make2d(X)
    y = make2d(y_)
    _fptype = getFPType(X)

    # only for dual_gap computation, it is not required for oneAPI
    # Data Analytics Library
    self._X = X
    self.n_features_in_ = X.shape[1]
    self._y = y

    # normalizing and centering
    X_offset = np.zeros(X.shape[1], dtype=X.dtype)
    X_scale = np.ones(X.shape[1], dtype=X.dtype)
    if y.ndim == 1:
        y_offset = X.dtype.type(0)
    else:
        y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    if sklearn_check_version("1.2"):
        _normalize = False
    else:
        _normalize = self._normalize
    if self.fit_intercept:
        X_offset = np.average(X, axis=0)
        if _normalize:
            if self.copy_X:
                X = np.copy(X) - X_offset
            else:
                X -= X_offset
            X, X_scale = normalize(X, axis=0, copy=False, return_norm=True)
            y_offset = np.average(y, axis=0)
            y = y - y_offset

    # only for compliance with Sklearn
    if (
        isinstance(self.precompute, np.ndarray)
        and self.fit_intercept
        and (
            not np.allclose(X_offset, np.zeros(X.shape[1]))
            or _normalize
            and not np.allclose(X_scale, np.ones(X.shape[1]))
        )
    ):
        warnings.warn(
            "Gram matrix was provided but X was centered"
            " to fit intercept, "
            "or X was normalized : recomputing Gram matrix.",
            UserWarning,
        )

    mse_alg = daal4py.optimization_solver_mse(
        numberOfTerms=X.shape[0], fptype=_fptype, method="defaultDense"
    )
    mse_alg.setup(X, y, None)

    cd_solver = daal4py.optimization_solver_coordinate_descent(
        function=mse_alg,
        fptype=_fptype,
        method="defaultDense",
        selection=self.selection,
        seed=0 if self.random_state is None else self.random_state,
        nIterations=self.max_iter,
        positive=self.positive,
        accuracyThreshold=self.tol,
    )

    # set warm_start
    if self.warm_start and hasattr(self, "coef_") and isinstance(self.coef_, np.ndarray):
        n_rows = y.shape[1]
        n_cols = X.shape[1] + 1
        inputArgument = np.zeros((n_rows, n_cols), dtype=_fptype)
        for i in range(n_rows):
            inputArgument[i][0] = self.intercept_ if (n_rows == 1) else self.intercept_[i]
            inputArgument[i][1:] = (
                self.coef_[:].copy(order="C")
                if (n_rows == 1)
                else self.coef_[i, :].copy(order="C")
            )
        cd_solver.setup(inputArgument)
    doUse_condition = self.copy_X is False or (
        self.fit_intercept and _normalize and self.copy_X
    )
    lasso_alg = daal4py.lasso_regression_training(
        fptype=_fptype,
        method="defaultDense",
        interceptFlag=(self.fit_intercept is True),
        dataUseInComputation="doUse" if doUse_condition else "doNotUse",
        lassoParameters=np.asarray(self.alpha, dtype=X.dtype).reshape((1, -1)),
        optimizationSolver=cd_solver,
    )
    try:
        if isinstance(self.precompute, np.ndarray):
            lasso_res = lasso_alg.compute(
                data=X, dependentVariables=y, gramMatrix=self.precompute
            )
        else:
            lasso_res = lasso_alg.compute(data=X, dependentVariables=y)
    except RuntimeError:
        return None

    # set coef_ and intersept_ results
    lasso_model = lasso_res.model
    self.daal_model_ = lasso_model

    # update coefficients if normalizing and centering
    if self.fit_intercept and _normalize:
        lasso_model.Beta[:, 1:] = lasso_model.Beta[:, 1:] / X_scale
        lasso_model.Beta[:, 0] = (
            y_offset - np.dot(X_offset, lasso_model.Beta[:, 1:].T)
        ).T

    coefs = lasso_model.Beta

    self.intercept_ = coefs[:, 0].copy(order="C")
    self.coef_ = coefs[:, 1:].copy(order="C")

    # only for compliance with Sklearn
    if y.shape[1] == 1:
        self.coef_ = np.ravel(self.coef_)
    self.intercept_ = np.ravel(self.intercept_)
    if self.intercept_.shape[0] == 1:
        self.intercept_ = self.intercept_[0]

    # set n_iter_
    n_iter = cd_solver.__get_result__().nIterations[0][0]
    if y.shape[1] == 1:
        self.n_iter_ = n_iter
    else:
        self.n_iter_ = np.full(y.shape[1], n_iter)

    # only for compliance with Sklearn
    if self.max_iter == n_iter + 1:
        warnings.warn(
            "Objective did not converge. You might want to "
            "increase the number of iterations.",
            ConvergenceWarning,
        )

    return self


def _daal4py_predict_lasso(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)

    lasso_palg = daal4py.lasso_regression_prediction(
        fptype=_fptype, method="defaultDense"
    )
    if self.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, "
            f"but Lasso is expecting "
            f"{self.n_features_in_} features as input"
        )
    lasso_res = lasso_palg.compute(X, self.daal_model_)

    res = lasso_res.prediction

    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)
    return res


def _fit(self, _X, _y, sample_weight=None, check_input=True):
    check_feature_names(self, _X, reset=True)
    if sklearn_check_version("1.2"):
        self._validate_params()
    elif sklearn_check_version("1.1"):
        check_scalar(
            self.alpha,
            "alpha",
            target_type=numbers.Real,
            min_val=0.0,
        )
        if self.alpha == 0:
            warnings.warn(
                "With alpha=0, this algorithm does not converge "
                "well. You are advised to use the LinearRegression "
                "estimator",
                stacklevel=2,
            )
        if isinstance(self.precompute, str):
            raise ValueError(
                "precompute should be one of True, False or array-like. Got %r"
                % self.precompute
            )
        check_scalar(
            self.l1_ratio,
            "l1_ratio",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=1.0,
        )
        if self.max_iter is not None:
            check_scalar(
                self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
            )
        check_scalar(self.tol, "tol", target_type=numbers.Real, min_val=0.0)
    # check X and y
    if check_input:
        X, y = check_X_y(
            _X,
            _y,
            copy=False,
            accept_sparse="csc",
            dtype=[np.float64, np.float32],
            multi_output=True,
            y_numeric=True,
        )
        y = check_array(_y, copy=False, dtype=X.dtype.type, ensure_2d=False)
    else:
        X, y = _X, _y

    if not sp.issparse(X):
        self.fit_shape_good_for_daal_ = (
            True if X.ndim <= 1 else True if X.shape[0] >= X.shape[1] else False
        )
    else:
        self.fit_shape_good_for_daal_ = False

    class_name = self.__class__.__name__
    class_inst = ElasticNet if class_name == "ElasticNet" else Lasso

    _function_name = f"sklearn.linear_model.{class_name}.fit"
    _patching_status = PatchingConditionsChain(_function_name)
    _dal_ready = _patching_status.and_conditions(
        [
            (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                self.fit_shape_good_for_daal_,
                "The shape of X does not satisfy oneDAL requirements: "
                "number of features > number of samples.",
            ),
            (
                X.dtype == np.float64 or X.dtype == np.float32,
                f"'{X.dtype}' X data type is not supported. "
                "Only np.float32 and np.float64 are supported.",
            ),
            (sample_weight is None, "Sample weights are not supported."),
        ]
    )
    _patching_status.write_log()

    if not _dal_ready:
        if hasattr(self, "daal_model_"):
            del self.daal_model_
        res_new = super(class_inst, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input
        )
        self._gap = res_new.dual_gap_
        return res_new
    self.n_iter_ = None
    self._gap = None

    if not check_input:
        # only for compliance with Sklearn,
        # this assert is not required for oneAPI Data
        # Analytics Library
        print(type(X), X.flags["F_CONTIGUOUS"])
        if isinstance(X, np.ndarray) and X.flags["F_CONTIGUOUS"] is False:
            # print(X.flags)
            raise ValueError("ndarray is not Fortran contiguous")

    if not sklearn_check_version("1.2"):
        self._normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=class_name
        )

    # only for pass tests
    # "check_estimators_fit_returns_self(readonly_memmap=True) and
    # check_regressors_train(readonly_memmap=True)
    if not X.flags.writeable:
        X = np.copy(X)
    if not y.flags.writeable:
        y = np.copy(y)

    if class_name == "ElasticNet":
        res = _daal4py_fit_enet(self, X, y, check_input=check_input)
    else:
        res = _daal4py_fit_lasso(self, X, y, check_input=check_input)
    if res is None:
        if hasattr(self, "daal_model_"):
            del self.daal_model_
        logging.info(_function_name + ": " + get_patch_message("sklearn_after_daal"))
        res_new = super(class_inst, self).fit(
            _X, _y, sample_weight=sample_weight, check_input=check_input
        )
        self._gap = res_new.dual_gap_
        return res_new
    return res


def _dual_gap(self):
    if self._gap is None:
        l1_reg = self.alpha * self.l1_ratio * self._X.shape[0]
        l2_reg = self.alpha * (1.0 - self.l1_ratio) * self._X.shape[0]
        n_targets = self._y.shape[1]

        if n_targets == 1:
            self._gap = self.tol + 1.0
            X_offset = np.average(self._X, axis=0)
            y_offset = np.average(self._y, axis=0)
            coef = np.reshape(self.coef_, (self.coef_.shape[0], 1))
            R = (self._y - y_offset) - np.dot((self._X - X_offset), coef)
            XtA = np.dot((self._X - X_offset).T, R) - l2_reg * coef
            R_norm2 = np.dot(R.T, R)
            coef_norm2 = np.dot(self.coef_, self.coef_)
            dual_norm_XtA = np.max(XtA) if self.positive else np.max(np.abs(XtA))
            if dual_norm_XtA > l1_reg:
                const = l1_reg / dual_norm_XtA
                A_norm2 = R_norm2 * (const**2)
                self._gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                self._gap = R_norm2
            l1_norm = np.sum(np.abs(self.coef_))
            tmp = l1_reg * l1_norm
            tmp -= const * np.dot(R.T, (self._y - y_offset))
            tmp += 0.5 * l2_reg * (1 + const**2) * coef_norm2
            self._gap += tmp
            self._gap = self._gap[0][0]
        else:
            self._gap = np.full(n_targets, self.tol + 1.0)
            X_offset = np.average(self._X, axis=0)
            y_offset = np.average(self._y, axis=0)
            for k in range(n_targets):
                R = (self._y[:, k] - y_offset[k]) - np.dot(
                    (self._X - X_offset), self.coef_[k, :].T
                )
                XtA = np.dot((self._X - X_offset).T, R) - l2_reg * self.coef_[k, :].T
                R_norm2 = np.dot(R.T, R)
                coef_norm2 = np.dot(self.coef_[k, :], self.coef_[k, :].T)
                dual_norm_XtA = np.max(XtA) if self.positive else np.max(np.abs(XtA))
                if dual_norm_XtA > l1_reg:
                    const = l1_reg / dual_norm_XtA
                    A_norm2 = R_norm2 * (const**2)
                    self._gap[k] = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    self._gap[k] = R_norm2
                l1_norm = np.sum(np.abs(self.coef_[k, :]))
                tmp = l1_reg * l1_norm
                tmp -= const * np.dot(R.T, (self._y[:, k] - y_offset[k]))
                tmp += 0.5 * l2_reg * (1 + const**2) * coef_norm2
                self._gap[k] += tmp
    return self._gap


@control_n_jobs(decorated_methods=["fit", "predict"])
class ElasticNet(ElasticNet_original):
    __doc__ = ElasticNet_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**ElasticNet_original._parameter_constraints}

        def __init__(
            self,
            alpha=1.0,
            l1_ratio=0.5,
            fit_intercept=True,
            precompute=False,
            max_iter=1000,
            copy_X=True,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
        ):
            super(ElasticNet, self).__init__(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                precompute=precompute,
                max_iter=max_iter,
                copy_X=copy_X,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                random_state=random_state,
                selection=selection,
            )

    else:

        def __init__(
            self,
            alpha=1.0,
            l1_ratio=0.5,
            fit_intercept=True,
            normalize="deprecated",
            precompute=False,
            max_iter=1000,
            copy_X=True,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
        ):
            super(ElasticNet, self).__init__(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                normalize=normalize,
                precompute=precompute,
                max_iter=max_iter,
                copy_X=copy_X,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                random_state=random_state,
                selection=selection,
            )

    def fit(self, X, y, sample_weight=None, check_input=True):
        return _fit(self, X, y, sample_weight=sample_weight, check_input=check_input)

    def predict(self, X):
        check_feature_names(self, X, reset=False)

        _X = check_array(
            X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float64, np.float32]
        )
        good_shape_for_daal = (
            True if _X.ndim <= 1 else True if _X.shape[0] >= _X.shape[1] else False
        )

        _patching_status = PatchingConditionsChain(
            "sklearn.linear_model.ElasticNet.predict"
        )
        _dal_ready = _patching_status.and_conditions(
            [
                (hasattr(self, "daal_model_"), "oneDAL model was not trained."),
                (not sp.issparse(_X), "X is sparse. Sparse input is not supported."),
                (
                    good_shape_for_daal,
                    "The shape of X does not satisfy oneDAL requirements: "
                    "number of features > number of samples.",
                ),
            ]
        )
        _patching_status.write_log()

        if not _dal_ready:
            return self._decision_function(X)
        return _daal4py_predict_enet(self, _X)

    @property
    def dual_gap_(self):
        return _dual_gap(self)

    @dual_gap_.setter
    def dual_gap_(self, value):
        self._gap = value

    @dual_gap_.deleter
    def dual_gap_(self):
        self._gap = None

    fit.__doc__ = ElasticNet_original.fit.__doc__
    predict.__doc__ = ElasticNet_original.predict.__doc__


@control_n_jobs(decorated_methods=["fit", "predict"])
class Lasso(Lasso_original):
    __doc__ = Lasso_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**Lasso_original._parameter_constraints}

        def __init__(
            self,
            alpha=1.0,
            fit_intercept=True,
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
        ):
            self.l1_ratio = 1.0
            super().__init__(
                alpha=alpha,
                fit_intercept=fit_intercept,
                precompute=precompute,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                random_state=random_state,
                selection=selection,
            )

    else:

        def __init__(
            self,
            alpha=1.0,
            fit_intercept=True,
            normalize="deprecated",
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
        ):
            self.l1_ratio = 1.0
            super().__init__(
                alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                precompute=precompute,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                random_state=random_state,
                selection=selection,
            )

    def fit(self, X, y, sample_weight=None, check_input=True):
        return _fit(self, X, y, sample_weight, check_input)

    def predict(self, X):
        check_feature_names(self, X, reset=False)
        _X = check_array(
            X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float64, np.float32]
        )
        good_shape_for_daal = (
            True if _X.ndim <= 1 else True if _X.shape[0] >= _X.shape[1] else False
        )

        _patching_status = PatchingConditionsChain("sklearn.linear_model.Lasso.predict")
        _dal_ready = _patching_status.and_conditions(
            [
                (hasattr(self, "daal_model_"), "oneDAL model was not trained."),
                (not sp.issparse(_X), "X is sparse. Sparse input is not supported."),
                (
                    good_shape_for_daal,
                    "The shape of X does not satisfy oneDAL requirements: "
                    "number of features > number of samples.",
                ),
            ]
        )
        _patching_status.write_log()

        if not _dal_ready:
            return self._decision_function(X)
        return _daal4py_predict_lasso(self, _X)

    @property
    def dual_gap_(self):
        return _dual_gap(self)

    @dual_gap_.setter
    def dual_gap_(self, value):
        self._gap = value

    @dual_gap_.deleter
    def dual_gap_(self):
        self._gap = None

    fit.__doc__ = Lasso_original.fit.__doc__
    predict.__doc__ = Lasso_original.predict.__doc__
