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
from sklearn.linear_model._logistic import _LOGISTIC_SOLVER_CONVERGENCE_MSG
from sklearn.linear_model._logistic import (
    LogisticRegression as LogisticRegression_original,
)
from sklearn.linear_model._logistic import (
    LogisticRegressionCV as LogisticRegressionCV_original,
)
from sklearn.linear_model._logistic import _check_solver
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils.optimize import _check_optimize_result, _newton_cg
from sklearn.utils.validation import check_is_fitted

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

if sklearn_check_version("1.7.1"):
    from sklearn.utils.fixes import _get_additional_lbfgs_options_dict
else:
    # From https://github.com/scikit-learn/scikit-learn/blob/760edca5fb5cc3538b98ebc55171806e2a6e3e84/sklearn/utils/fixes.py#L408
    # This should be removed if SciPy>=1.15 becomes the minimum required at some point
    def _get_additional_lbfgs_options_dict(k, v):
        return {k: v}


from sklearn.linear_model._logistic import _logistic_regression_path as lr_path_original
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


# This code is a patch for sklearn 1.8, which is related to https://github.com/scikit-learn/scikit-learn/pull/32073
# where the multi_class keyword is deprecated and this aspect is removed.
def _check_multi_class(multi_class, solver, n_classes):
    """Computes the multi class type, either "multinomial" or "ovr".
    For `n_classes` > 2 and a solver that supports it, returns "multinomial".
    For all other cases, in particular binary classification, return "ovr".
    """
    if multi_class == "auto":
        if solver in ("liblinear",):
            multi_class = "ovr"
        elif n_classes > 2:
            multi_class = "multinomial"
        else:
            multi_class = "ovr"
    if multi_class == "multinomial" and solver in ("liblinear",):
        raise ValueError("Solver %s does not support a multinomial backend." % solver)
    return multi_class


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
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    n_threads=1,
):

    # Comment 2025-08-04: this file might have dead code paths from unsupported solvers.
    # It appears to have initially been a copy-paste of scikit-learn with a few additions
    # for varying levels of offloading to oneDAL, but later on a check was added that
    # calls 'lr_path_original' early on when it won't end up offloading anything to oneDAL.
    # Some parts of the file have been selectively updated since the initial copy-paste
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
        y_bin = (y == pos_class).astype(X.dtype)
        w0 = np.zeros(n_features + 1, dtype=X.dtype)
    else:
        Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
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
        w0 = w0.ravel(order="C")
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
            # Note: this uses more correction pairs than the implementation in scikit-learn,
            # which means better approximation of the Hessian at the expense of slower updates.
            # This is beneficial for high-dimensional convex problems without bound constraints
            # like the logistic regression being fitted here. For larger problems with sparse
            # data (currently not supported), it might benefit from increasing the number further.
            opt_res = optimize.minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=extra_args,
                options={
                    "maxiter": max_iter,
                    "maxcor": 50,
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


def daal4py_fit_cv(self, X, y, sample_weight=None, **params):
    which, what = logistic_module, "_logistic_regression_path"
    replacer = logistic_regression_path_cv
    try:
        setattr(which, what, replacer)
        clf = LogisticRegressionCV_original.fit(self, X, y, sample_weight, **params)
    finally:
        setattr(which, what, lr_path_original)
    return clf


def daal4py_predict(self, X, resultsToEvaluate):
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
    _dal_ready = _patching_status.and_conditions(
        [
            (
                not ((not isinstance(X, np.ndarray)) and hasattr(X, "__dlpack__")),
                "Array API inputs not supported.",
            )
        ]
    )
    if not _dal_ready:
        return getattr(LogisticRegression_original, _function_name)(self, X)

    check_is_fitted(self)
    check_feature_names(self, X, reset=False)
    X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    if _function_name != "predict":
        multi_class = getattr(self, "multi_class", "auto")
        _patching_status.and_conditions(
            [
                (
                    self.classes_.size == 2
                    or _check_multi_class(
                        multi_class if multi_class != "deprecated" else "auto",
                        self.solver,
                        self.classes_.size,
                    )
                    != "ovr",
                    f"selected multiclass option is not supported for n_classes > 2.",
                ),
                (
                    not (self.classes_.size == 2 and multi_class == "multinomial"),
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


def logistic_regression_path(*args, **kwargs):
    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LogisticRegression.fit"
    )
    return logistic_regression_path_internal(_patching_status, *args, **kwargs)


def logistic_regression_path_cv(*args, **kwargs):
    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LogisticRegressionCV.fit"
    )
    return logistic_regression_path_internal(_patching_status, *args, **kwargs)


def logistic_regression_path_internal(_patching_status, *args, **kwargs):
    _dal_ready = _patching_status.and_conditions(
        [
            (
                kwargs["solver"] in ["lbfgs", "newton-cg"],
                f"'{kwargs['solver']}' solver is not supported. "
                "Only 'lbfgs' and 'newton-cg' solvers are supported.",
            ),
            (not sparse.issparse(args[0]), "X is sparse. Sparse input is not supported."),
            (
                not (
                    (not isinstance(args[0], np.ndarray))
                    and hasattr(args[0], "__dlpack__")
                ),
                "Array API inputs not supported.",
            ),
            (kwargs["sample_weight"] is None, "Sample weights are not supported."),
            (kwargs["class_weight"] is None, "Class weights are not supported."),
            (
                kwargs["penalty"]
                in (["l2", "deprecated"] if sklearn_check_version("1.8") else ["l2"]),
                "Penalties other than l2 are not supported.",
            ),
            (not kwargs["l1_ratio"], "L1 regularization is not supported."),
            (
                not (kwargs["solver"] == "newton-cg" and not kwargs["fit_intercept"]),
                "'newton-cg' solver without intercept is not supported.",
            ),
        ]
    )
    if not _dal_ready:
        _patching_status.write_log()
        return lr_path_original(*args, **kwargs)

    if sklearn_check_version("1.8"):
        kwargs.pop("classes", None)
        res = __logistic_regression_path(*(args[:2]), **kwargs)
    else:
        res = __logistic_regression_path(*args, **kwargs)

    _patching_status.write_log()
    return res


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


@control_n_jobs(
    decorated_methods=["fit", "predict", "predict_proba", "predict_log_proba"]
)
class LogisticRegressionCV(LogisticRegressionCV_original):

    def fit(self, X, y, sample_weight=None, **params):
        return daal4py_fit_cv(self, X, y, sample_weight, **params)

    def predict(self, X):
        return daal4py_predict(self, X, "computeClassLabels")

    def predict_log_proba(self, X):
        return daal4py_predict(self, X, "computeClassLogProbabilities")

    def predict_proba(self, X):
        return daal4py_predict(self, X, "computeClassProbabilities")

    __doc__ = LogisticRegressionCV_original.__doc__
    fit.__doc__ = LogisticRegressionCV_original.fit.__doc__
    predict.__doc__ = LogisticRegressionCV_original.predict.__doc__
    predict_log_proba.__doc__ = LogisticRegressionCV_original.predict_log_proba.__doc__
    predict_proba.__doc__ = LogisticRegressionCV_original.predict_proba.__doc__
