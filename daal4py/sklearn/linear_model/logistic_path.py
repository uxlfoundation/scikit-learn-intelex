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
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1', 'l2', or 'elasticnet'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float or None, optional (default=None)
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    _patching_status = PatchingConditionsChain(
        "sklearn.linear_model.LogisticRegression.fit"
    )
    # TODO: remove this fallback workaround after
    # logistic path is reworked to align with sklearn 1.2
    _dal_ready = _patching_status.and_conditions(
        [
            (
                not (sklearn_check_version("1.2") and solver == "newton-cholesky"),
                f"'{solver}' solver is not supported. "
                "Only 'lbfgs' and 'newton-cg' solvers are supported.",
            )
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

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        if sklearn_check_version("1.1"):
            X = check_array(
                X,
                accept_sparse="csr",
                dtype=np.float64,
                accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
            )
        else:
            X = check_array(
                X,
                accept_sparse="csr",
                dtype=np.float64,
                accept_large_sparse=solver != "liblinear",
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
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)
    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if (
        isinstance(class_weight, dict) or multi_class == "multinomial"
    ) and not _dal_ready:
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        if not np.allclose(class_weight_, np.ones_like(class_weight_)):
            sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == "ovr":
        y_bin = np.ones(y.shape, dtype=X.dtype)

        if sklearn_check_version("1.1"):
            mask = y == pos_class
            y_bin = np.ones(y.shape, dtype=X.dtype)
            # for compute_class_weight

            if solver == "liblinear" or (
                not sklearn_check_version("1.6") and solver not in ["lbfgs", "newton-cg"]
            ):
                mask_classes = np.array([-1, 1])
                y_bin[~mask] = -1.0
            else:
                # HalfBinomialLoss, used for those solvers, represents y in [0, 1] instead
                # of in [-1, 1].
                mask_classes = np.array([0, 1])
                y_bin[~mask] = 0.0
        else:
            mask_classes = np.array([-1, 1])
            mask = y == pos_class
            y_bin[~mask] = -1.0
            # for compute_class_weight

        if class_weight == "balanced" and not _dal_ready:
            class_weight_ = compute_class_weight(
                class_weight, classes=mask_classes, y=y_bin
            )
            if not np.allclose(class_weight_, np.ones_like(class_weight_)):
                sample_weight *= class_weight_[le.fit_transform(y_bin)]

        if _dal_ready:
            w0 = np.zeros(n_features + 1, dtype=X.dtype)
            y_bin[~mask] = 0.0
        else:
            w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)

    else:
        if sklearn_check_version("1.1"):
            if sklearn_check_version("1.6"):
                solver_list = ["sag", "saga", "lbfgs", "newton-cg", "newton-cholesky"]
            else:
                solver_list = ["sag", "saga", "lbfgs", "newton-cg"]
            if solver in solver_list:
                # SAG, lbfgs and newton-cg multinomial solvers need LabelEncoder,
                # not LabelBinarizer, i.e. y as a 1d-array of integers.
                # LabelEncoder also saves memory compared to LabelBinarizer, especially
                # when n_classes is large.
                if _dal_ready:
                    Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
                else:
                    le = LabelEncoder()
                    Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
            else:
                # For liblinear solver, apply LabelBinarizer, i.e. y is one-hot encoded.
                lbin = LabelBinarizer()
                Y_multi = lbin.fit_transform(y)
                if Y_multi.shape[1] == 1:
                    Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            if solver not in ["sag", "saga"]:
                if _dal_ready:
                    Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
                else:
                    lbin = LabelBinarizer()
                    Y_multi = lbin.fit_transform(y)
                    if Y_multi.shape[1] == 1:
                        Y_multi = np.hstack([1 - Y_multi, Y_multi])
            else:
                # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
                le = LabelEncoder()
                Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        if _dal_ready:
            w0 = np.zeros((classes.size, n_features + 1), order="C", dtype=X.dtype)
        else:
            w0 = np.zeros(
                (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
            )

    # Adoption of https://github.com/scikit-learn/scikit-learn/pull/26721
    if solver in ["lbfgs", "newton-cg", "newton-cholesky"]:
        sw_sum = len(X) if sample_weight is None else np.sum(sample_weight)

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == "ovr":
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape "
                    "%d or %d" % (coef.size, n_features, w0.size)
                )
            if _dal_ready:
                w0[-coef.size :] = (
                    np.roll(coef, 1, -1) if coef.size != n_features else coef
                )
            else:
                w0[: coef.size] = coef
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

            if _dal_ready:
                w0[:, -coef.shape[1] :] = (
                    np.roll(coef, 1, -1) if coef.shape[1] != n_features else coef
                )
            else:
                if n_classes == 1:
                    w0[0, : coef.shape[1]] = -coef
                    w0[1, : coef.shape[1]] = coef
                else:
                    w0[:, : coef.shape[1]] = coef

    C_daal_multiplier = 1
    # commented out because this is Py3 feature
    # def _map_to_binary_logistic_regression():
    #    nonlocal C_daal_multiplier
    #    nonlocal w0
    #    C_daal_multiplier = 2
    #    w0 *= 2

    if multi_class == "multinomial":
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if sklearn_check_version("1.6"):
            solver_list = ["lbfgs", "newton-cg", "newton-cholesky"]
        else:
            solver_list = ["lbfgs", "newton-cg"]
        if solver in solver_list:
            if _dal_ready and classes.size == 2:
                w0 = w0[-1:, :]
            if sklearn_check_version("1.1"):
                w0 = w0.ravel(order="F")
            else:
                w0 = w0.ravel()
        target = Y_multi
        loss = None
        if sklearn_check_version("1.1"):
            loss = LinearModelLoss(
                base_loss=HalfMultinomialLoss(n_classes=classes.size),
                fit_intercept=fit_intercept,
            )
        if solver == "lbfgs":
            if _dal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_and_grad
            else:
                if sklearn_check_version("1.1") and loss is not None:
                    func = loss.loss_gradient
                else:

                    def func(x, *args):
                        return _multinomial_loss_grad(x, *args)[0:2]

        elif solver == "newton-cg":
            if _dal_ready:
                if classes.size == 2:
                    # _map_to_binary_logistic_regression()
                    C_daal_multiplier = 2
                    w0 *= 2
                    daal_extra_args_func = _daal4py_logistic_loss_extra_args
                else:
                    daal_extra_args_func = _daal4py_cross_entropy_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                if sklearn_check_version("1.1") and loss is not None:
                    func = loss.loss
                    grad = loss.gradient
                    hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
                else:

                    def func(x, *args):
                        return _multinomial_loss(x, *args)[0]

                    def grad(x, *args):
                        return _multinomial_loss_grad(x, *args)[1]

                    hess = _multinomial_grad_hess
        warm_start_sag = {"coef": w0.T}
    else:
        target = y_bin
        if solver == "lbfgs":
            if _dal_ready:
                func = _daal4py_loss_and_grad
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
            else:
                if sklearn_check_version("1.1"):
                    loss = LinearModelLoss(
                        base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
                    )
                    func = loss.loss_gradient
                else:
                    func = _logistic_loss_and_grad
        elif solver == "newton-cg":
            if _dal_ready:
                daal_extra_args_func = _daal4py_logistic_loss_extra_args
                func = _daal4py_loss_
                grad = _daal4py_grad_
                hess = _daal4py_grad_hess_
            else:
                if sklearn_check_version("1.1"):
                    loss = LinearModelLoss(
                        base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
                    )
                    func = loss.loss
                    grad = loss.gradient
                    hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
                else:
                    func = _logistic_loss

                    def grad(x, *args):
                        return _logistic_loss_and_grad(x, *args)[1]

                    hess = _logistic_grad_hess
        warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == "lbfgs":
            if _dal_ready:
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
            else:
                if sklearn_check_version("1.1"):
                    l2_reg_strength = 1.0 / (C * sw_sum)
                    extra_args = (X, target, sample_weight, l2_reg_strength, n_threads)
                else:
                    extra_args = (X, target, 1.0 / (C * sw_sum), sample_weight)

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
                    "iprint": iprint,
                    "gtol": tol,
                    "ftol": 64 * np.finfo(float).eps,
                },
            )
            n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = opt_res.x, opt_res.fun
            if _dal_ready and C_daal_multiplier == 2:
                w0 /= 2
        elif solver == "newton-cg":
            if _dal_ready:

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
            else:
                if sklearn_check_version("1.1"):
                    l2_reg_strength = 1.0 / (C * sw_sum)
                    args = (X, target, sample_weight, l2_reg_strength, n_threads)
                else:
                    args = (X, target, 1.0 / (C * sw_sum), sample_weight)

                w0, n_iter_i = _newton_cg(
                    hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
                )
        elif solver == "liblinear":
            (
                coef_,
                intercept_,
                n_iter_i,
            ) = _fit_liblinear(
                X,
                target,
                C,
                fit_intercept,
                intercept_scaling,
                None,
                penalty,
                dual,
                verbose,
                max_iter,
                tol,
                random_state,
                sample_weight=sample_weight,
            )
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ["sag", "saga"]:
            if multi_class == "multinomial":
                target = target.astype(X.dtype, copy=False)
                loss = "multinomial"
            else:
                loss = "log"
            # alpha is for L2-norm, beta is for L1-norm
            if penalty == "l1":
                alpha = 0.0
                beta = 1.0 / C
            elif penalty == "l2":
                alpha = 1.0 / C
                beta = 0.0
            else:  # Elastic-Net penalty
                alpha = (1.0 / C) * (1 - l1_ratio)
                beta = (1.0 / C) * l1_ratio

            w0, n_iter_i, warm_start_sag = sag_solver(
                X,
                target,
                sample_weight,
                loss,
                alpha,
                beta,
                max_iter,
                tol,
                verbose,
                random_state,
                False,
                max_squared_sum,
                warm_start_sag,
                is_saga=(solver == "saga"),
            )

        else:
            raise ValueError(
                "solver must be one of {'liblinear', 'lbfgs', "
                "'newton-cg', 'sag'}, got '%s' instead" % solver
            )

        if multi_class == "multinomial":
            if _dal_ready:
                if classes.size == 2:
                    multi_w0 = w0[np.newaxis, :]
                else:
                    multi_w0 = np.reshape(w0, (classes.size, -1))
            else:
                n_classes = max(2, classes.size)
                if sklearn_check_version("1.1"):
                    if sklearn_check_version("1.6"):
                        solver_list = ["lbfgs", "newton-cg", "newton-cholesky"]
                    else:
                        solver_list = ["lbfgs", "newton-cg"]
                    if solver in solver_list:
                        multi_w0 = np.reshape(w0, (n_classes, -1), order="F")
                    else:
                        multi_w0 = w0
                else:
                    multi_w0 = np.reshape(w0, (n_classes, -1))
                if n_classes == 2:
                    multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0.copy())
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    if _dal_ready:
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
