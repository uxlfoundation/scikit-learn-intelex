# ==============================================================================
# Copyright 2014 Intel Corporation
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
import scipy.optimize as optimize
import scipy.sparse as sparse
import sklearn.linear_model._logistic as logistic_module
from sklearn.linear_model._sag import sag_solver
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    compute_class_weight,
)
from sklearn.utils.fixes import _get_additional_lbfgs_options_dict
from sklearn.utils.optimize import _check_optimize_result, _newton_cg
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

import daal4py as d4p

from .._n_jobs_support import control_n_jobs
from .._utils import PatchingConditionsChain, getFPType, sklearn_check_version
from ..utils.validation import check_feature_names
from .logistic_loss import (
    _daal4py_cross_entropy_loss_extra_args,
    _daal4py_grad_,
    _daal4py_grad_hess_,
    _daal4py_logistic_loss_extra_args,
    _daal4py_loss_,
    _daal4py_loss_and_grad,
)

if sklearn_check_version("1.1"):
    from sklearn._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
    from sklearn.linear_model._linear_loss import LinearModelLoss
    from sklearn.linear_model._logistic import _LOGISTIC_SOLVER_CONVERGENCE_MSG
    from sklearn.linear_model._logistic import (
        LogisticRegression as LogisticRegression_original,
    )
    from sklearn.linear_model._logistic import (
        _check_multi_class,
        _check_solver,
        _fit_liblinear,
    )
else:
    from sklearn.linear_model._logistic import (
        _check_solver,
        _check_multi_class,
        _fit_liblinear,
        _logistic_loss_and_grad,
        _logistic_loss,
        _logistic_grad_hess,
        _multinomial_loss,
        _multinomial_loss_grad,
        _multinomial_grad_hess,
        _LOGISTIC_SOLVER_CONVERGENCE_MSG,
        LogisticRegression as LogisticRegression_original,
    )

from sklearn.linear_model._logistic import _logistic_regression_path as lr_path_original
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


# Code adapted from sklearn.linear_model.logistic version 0.21
def __logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="warn",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    n_threads=1,
):
    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LogisticRegression.fit"
    )
    _dal_ready = _patching_status.and_conditions(
        [
            (
                solver in ["lbfgs", "newton-cg"],
                f"'{solver}' solver is not supported. "
                "Only 'lbfgs' and 'newton-cg' solvers are supported.",
            ),
            (not sparse.issparse(X), "X is sparse. Sparse input is not supported."),
            (sample_weight is None, "Sample weights are not supported."),
            (class_weight is None, "Class weights are not supported."),
        ]
    )
    if not _dal_ready:
        _patching_status.write_log()
        return lr_path_original(
            X,
            y,
            pos_class=pos_class,
            Cs=Cs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            solver=solver,
            coef=coef,
            class_weight=class_weight,
            dual=dual,
            penalty=penalty,
            intercept_scaling=intercept_scaling,
            multi_class=multi_class,
            random_state=random_state,
            check_input=check_input,
            max_squared_sum=max_squared_sum,
            sample_weight=sample_weight,
            l1_ratio=l1_ratio,
            n_threads=n_threads,
        )

    # Comment 2025-08-04: this file might have dead code paths from unsupported solvers.
    # It appears to have initially been a copy-paste of scikit-learn with a few additions
    # for varying levels of offloading to oneDAL, but later on the check above was added that
    # calls 'lr_path_original' early on when it won't end up doing offloading anything to
    # oneDAL. Some parts of the file have been selectively updated since the initial copy-paste
    # to reflect newer additions to sklearn, but they are not synch. The rest of the file
    # remains as it was before the early offload conditions, so some sections might have
    # become unreachable.

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            accept_large_sparse=False,
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != "multinomial":
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    le = LabelEncoder()

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == "ovr":
        y_bin = np.ones(y.shape, dtype=X.dtype)

        if sklearn_check_version("1.1"):
            mask = y == pos_class
            y_bin = np.ones(y.shape, dtype=X.dtype)

            y_bin[~mask] = 0.0
        else:
            mask = y == pos_class
            y_bin[~mask] = -1.0

        w0 = np.zeros(n_features + 1, dtype=X.dtype)
        y_bin[~mask] = 0.0

    else:
        _multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros((classes.size, n_features + 1), order="C", dtype=X.dtype)

    # Adoption of https://github.com/scikit-learn/scikit-learn/pull/26721
    sw_sum = len(X)

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == "ovr":
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape "
                    "%d or %d" % (coef.size, n_features, w0.size)
                )
            w0[-coef.size :] = np.roll(coef, 1, -1) if coef.size != n_features else coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if coef.shape[0] != n_classes or coef.shape[1] not in (
                n_features,
                n_features + 1,
            ):
                raise ValueError(
                    "Initialization coef is of shape (%d, %d), expected "
                    "shape (%d, %d) or (%d, %d)"
                    % (
                        coef.shape[0],
                        coef.shape[1],
                        classes.size,
                        n_features,
                        classes.size,
                        n_features + 1,
                    )
                )

            w0[:, -coef.shape[1] :] = (
                np.roll(coef, 1, -1) if coef.shape[1] != n_features else coef
            )

    C_daal_multiplier = 1

    if multi_class == "multinomial":
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if classes.size == 2:
            w0 = w0[-1:, :]
        if sklearn_check_version("1.1"):
            if _dal_ready and coef is not None:
                w0 = w0.ravel(order="C")
            else:
                w0 = w0.ravel(order="F")
        else:
            w0 = w0.ravel()
        target = Y_multi

        # Note: scikit-learn does a theoretically incorrect procedure when using
        # multi_class='multinomial' with two classes. This converts the problem
        # into an equivalent problem for binary logistic regression.
        if solver == "lbfgs":
            if classes.size == 2:
                C_daal_multiplier = 2
                w0 *= 2
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
            else:
                daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
            func = _daal4py_loss_and_grad

        elif solver == "newton-cg":
            if classes.size == 2:
                C_daal_multiplier = 2
                w0 *= 2
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
            else:
                daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
            func = _daal4py_loss_
            grad = _daal4py_grad_
            hess = _daal4py_grad_hess_
        warm_start_sag = {"coef": w0.T}
    else:
        target = y_bin
        if solver == "lbfgs":
            func = _daal4py_loss_and_grad
            daal_extra_args_func = _daal4py_logistic_loss_extra_args
        elif solver == "newton-cg":
            daal_extra_args_func = _daal4py_logistic_loss_extra_args
            func = _daal4py_loss_
            grad = _daal4py_grad_
            hess = _daal4py_grad_hess_
        warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == "lbfgs":
            extra_args = daal_extra_args_func(
                classes.size,
                w0,
                X,
                target,
                0.0,
                1.0 / (2 * C * C_daal_multiplier * sw_sum),
                fit_intercept,
                value=True,
                gradient=True,
                hessian=False,
            )

            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)
            ]
            opt_res = optimize.minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=extra_args,
                options={
                    "maxiter": max_iter,
                    "maxls": 50,
                    "gtol": tol,
                    "ftol": 64 * np.finfo(float).eps,
                    **_get_additional_lbfgs_options_dict("iprint", iprint),
                },
            )
            n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = opt_res.x, opt_res.fun
            if C_daal_multiplier == 2:
                w0 /= 2
        elif solver == "newton-cg":

            def make_ncg_funcs(f, value=False, gradient=False, hessian=False):
                daal_penaltyL2 = 1.0 / (2 * C * C_daal_multiplier * sw_sum)
                _obj_, X_, y_, n_samples = daal_extra_args_func(
                    classes.size,
                    w0,
                    X,
                    target,
                    0.0,
                    daal_penaltyL2,
                    fit_intercept,
                    value=value,
                    gradient=gradient,
                    hessian=hessian,
                )

                def _func_(x, *args):
                    return f(x, _obj_, *args)

                return _func_, (X_, y_, n_samples, daal_penaltyL2)

            loss_func, extra_args = make_ncg_funcs(func, value=True)
            grad_func, _ = make_ncg_funcs(grad, gradient=True)
            grad_hess_func, _ = make_ncg_funcs(hess, gradient=True)
            w0, n_iter_i = _newton_cg(
                grad_hess_func,
                loss_func,
                grad_func,
                w0,
                args=extra_args,
                maxiter=max_iter,
                tol=tol,
            )
            if C_daal_multiplier == 2:
                w0 /= 2

        else:
            raise ValueError(
                "solver must be one of {'lbfgs', 'newton-cg'}, got '%s' instead" % solver
            )

        if multi_class == "multinomial":
            if classes.size == 2:
                multi_w0 = w0[np.newaxis, :]
            else:
                multi_w0 = np.reshape(w0, (classes.size, -1))
            coefs.append(multi_w0.copy())
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    if fit_intercept:
        for i, ci in enumerate(coefs):
            coefs[i] = np.roll(ci, -1, -1)
    else:
        for i, ci in enumerate(coefs):
            coefs[i] = np.delete(ci, 0, axis=-1)

    _patching_status.write_log()

    return np.array(coefs), np.array(Cs), n_iter


def daal4py_fit(self, X, y, sample_weight=None):
    which, what = logistic_module, "_logistic_regression_path"
    replacer = logistic_regression_path
    descriptor = getattr(which, what, None)
    setattr(which, what, replacer)
    try:
        clf = LogisticRegression_original.fit(self, X, y, sample_weight)
    finally:
        setattr(which, what, descriptor)
    return clf


def daal4py_predict(self, X, resultsToEvaluate):
    check_is_fitted(self)
    check_feature_names(self, X, reset=False)
    X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    if resultsToEvaluate == "computeClassLabels":
        _function_name = "predict"
    elif resultsToEvaluate == "computeClassProbabilities":
        _function_name = "predict_proba"
    elif resultsToEvaluate == "computeClassLogProbabilities":
        _function_name = "predict_log_proba"
    else:
        raise ValueError(
            "resultsToEvaluate must be in [computeClassLabels, \
            computeClassProbabilities, computeClassLogProbabilities]"
        )

    _patching_status = PatchingConditionsChain(
        f"sklearn.linear_model.LogisticRegression.{_function_name}"
    )
    if _function_name != "predict":
        _patching_status.and_conditions(
            [
                (
                    self.classes_.size == 2
                    or logistic_module._check_multi_class(
                        self.multi_class if self.multi_class != "deprecated" else "auto",
                        self.solver,
                        self.classes_.size,
                    )
                    != "ovr",
                    f"selected multiclass option is not supported for n_classes > 2.",
                ),
                (
                    not (self.classes_.size == 2 and self.multi_class == "multinomial"),
                    "multi_class='multinomial' not supported with binary data",
                ),
            ],
        )

    _dal_ready = _patching_status.and_conditions(
        [
            (not sparse.issparse(X), "X is sparse. Sparse input is not supported."),
            (
                not sparse.issparse(self.coef_),
                "self.coef_ is sparse. Sparse coefficients are not supported.",
            ),
            (fptype is not None, "Unable to get dtype."),
        ]
    )

    _patching_status.write_log()
    if _dal_ready:
        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but LogisticRegression is expecting {n_features} features as input"
            )
        builder = d4p.logistic_regression_model_builder(X.shape[1], len(self.classes_))
        builder.set_beta(self.coef_, self.intercept_)
        predict = d4p.logistic_regression_prediction(
            nClasses=len(self.classes_),
            fptype=fptype,
            method="defaultDense",
            resultsToEvaluate=resultsToEvaluate,
        )
        res = predict.compute(X, builder.model)
        if resultsToEvaluate == "computeClassLabels":
            res = res.prediction
            if (
                not np.array_equal(self.classes_, np.arange(0, len(self.classes_)))
                or self.classes_.dtype != X.dtype
            ):
                res = self.classes_.take(np.asarray(res, dtype=np.intp))
        elif resultsToEvaluate == "computeClassProbabilities":
            res = res.probabilities
        elif resultsToEvaluate == "computeClassLogProbabilities":
            res = res.logProbabilities
        else:
            raise ValueError(
                "resultsToEvaluate must be in [computeClassLabels, \
                computeClassProbabilities, computeClassLogProbabilities]"
            )
        if res.shape[1] == 1:
            res = np.ravel(res)
        return res

    if resultsToEvaluate == "computeClassLabels":
        return LogisticRegression_original.predict(self, X)
    if resultsToEvaluate == "computeClassProbabilities":
        return LogisticRegression_original.predict_proba(self, X)
    if resultsToEvaluate == "computeClassLogProbabilities":
        return LogisticRegression_original.predict_log_proba(self, X)


def logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    n_threads=1,
):
    if sklearn_check_version("1.1"):
        return __logistic_regression_path(
            X,
            y,
            pos_class=pos_class,
            Cs=Cs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            solver=solver,
            coef=coef,
            class_weight=class_weight,
            dual=dual,
            penalty=penalty,
            intercept_scaling=intercept_scaling,
            multi_class=multi_class,
            random_state=random_state,
            check_input=check_input,
            max_squared_sum=max_squared_sum,
            sample_weight=sample_weight,
            l1_ratio=l1_ratio,
            n_threads=n_threads,
        )
    return __logistic_regression_path(
        X,
        y,
        pos_class=pos_class,
        Cs=Cs,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        solver=solver,
        coef=coef,
        class_weight=class_weight,
        dual=dual,
        penalty=penalty,
        intercept_scaling=intercept_scaling,
        multi_class=multi_class,
        random_state=random_state,
        check_input=check_input,
        max_squared_sum=max_squared_sum,
        sample_weight=sample_weight,
        l1_ratio=l1_ratio,
    )


@control_n_jobs(
    decorated_methods=["fit", "predict", "predict_proba", "predict_log_proba"]
)
class LogisticRegression(LogisticRegression_original):
    __doc__ = LogisticRegression_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **LogisticRegression_original._parameter_constraints
        }

    def __init__(
        self,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y, sample_weight=None):
        check_feature_names(self, X, reset=True)
        if sklearn_check_version("1.2"):
            self._validate_params()
        return daal4py_fit(self, X, y, sample_weight)

    def predict(self, X):
        return daal4py_predict(self, X, "computeClassLabels")

    def predict_log_proba(self, X):
        return daal4py_predict(self, X, "computeClassLogProbabilities")

    def predict_proba(self, X):
        return daal4py_predict(self, X, "computeClassProbabilities")

    fit.__doc__ = LogisticRegression_original.fit.__doc__
    predict.__doc__ = LogisticRegression_original.predict.__doc__
    predict_log_proba.__doc__ = LogisticRegression_original.predict_log_proba.__doc__
    predict_proba.__doc__ = LogisticRegression_original.predict_proba.__doc__
