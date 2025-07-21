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

import numpy as np

import daal4py

from .._utils import getFPType, make2d


def _resultsToCompute_string(value=True, gradient=True, hessian=False):
    results_needed = []
    if value:
        results_needed.append("value")
    if gradient:
        results_needed.append("gradient")
    if hessian:
        results_needed.append("hessian")

    return "|".join(results_needed)


def _daal4py_logistic_loss_extra_args(
    nClasses_unused,
    beta,
    X,
    y,
    l1=0.0,
    l2=0.0,
    fit_intercept=True,
    value=True,
    gradient=True,
    hessian=False,
):
    X = make2d(X)
    nSamples, nFeatures = X.shape

    y = make2d(y)
    beta = make2d(beta)
    n = X.shape[0]

    results_to_compute = _resultsToCompute_string(
        value=value, gradient=gradient, hessian=hessian
    )

    objective_function_algorithm_instance = daal4py.optimization_solver_logistic_loss(
        numberOfTerms=n,
        fptype=getFPType(X),
        method="defaultDense",
        interceptFlag=fit_intercept,
        penaltyL1=l1 / n,
        penaltyL2=l2 / n,
        resultsToCompute=results_to_compute,
    )
    objective_function_algorithm_instance.setup(X, y, beta)

    return (objective_function_algorithm_instance, X, y, n)


def _daal4py_cross_entropy_loss_extra_args(
    nClasses,
    beta,
    X,
    y,
    l1=0.0,
    l2=0.0,
    fit_intercept=True,
    value=True,
    gradient=True,
    hessian=False,
):
    X = make2d(X)
    nSamples, nFeatures = X.shape
    y = make2d(y)
    beta = make2d(beta)
    n = X.shape[0]

    results_to_compute = _resultsToCompute_string(
        value=value, gradient=gradient, hessian=hessian
    )

    objective_function_algorithm_instance = (
        daal4py.optimization_solver_cross_entropy_loss(
            nClasses=nClasses,
            numberOfTerms=n,
            fptype=getFPType(X),
            method="defaultDense",
            interceptFlag=fit_intercept,
            penaltyL1=l1 / n,
            penaltyL2=l2 / n,
            resultsToCompute=results_to_compute,
        )
    )
    objective_function_algorithm_instance.setup(X, y, beta)

    return (objective_function_algorithm_instance, X, y, n)


# used by LBFGS solver
def _daal4py_loss_and_grad(beta, objF_instance, X, y, n):
    beta_ = make2d(beta)
    res = objF_instance.compute(X, y, beta_)
    gr = res.gradientIdx.ravel()
    gr = gr * n  # force copy
    v = res.valueIdx[0, 0]
    v *= n
    return (v, gr)


# used by Newton CG method
def _daal4py_loss_(beta, objF_instance, X, y, n, l2_unused):
    beta_ = make2d(beta)
    if beta_.shape[1] != 1 and beta_.shape[0] == 1:
        beta_ = beta_.T
    res = objF_instance.compute(X, y, beta_)
    v = res.valueIdx[0, 0]
    v *= n
    return v


def _daal4py_grad_(beta, objF_instance, X, y, n, l2_unused):
    beta_ = make2d(beta)
    if beta_.shape[1] != 1 and beta_.shape[0] == 1:
        beta_ = beta_.T
    res = objF_instance.compute(X, y, beta_)
    # Copy is needed for newton-cg to work correctly
    # Otherwise evaluation at nearby point in line search
    # overwrites gradient at the starting point
    gr = res.gradientIdx.ravel()
    gr = gr * n  # force copy
    return gr


def _daal4py_grad_hess_(beta, objF_instance, X, y, n, l2):
    beta_ = make2d(beta)
    if beta_.shape[1] != 1 and beta_.shape[0] == 1:
        beta_ = beta_.T
    res = objF_instance.compute(X, y, beta_)
    gr = res.gradientIdx.ravel()
    gr = gr * n

    if isinstance(objF_instance, daal4py.optimization_solver_logistic_loss):
        # dealing with binary logistic regression
        # pp - array of probabilities for class=1, shape=(nSamples,)
        pp = beta_[0, 0] + np.dot(X, beta_[1:, 0])
        y2 = -1 + 2 * y[:, 0]
        pp *= y2
        np.exp(pp, out=pp)
        pp += 1
        np.reciprocal(pp, out=pp)
        np.square(pp, out=pp)
        del y2

        def hessp(v):
            pp0 = pp * (v[0] + np.dot(X, v[1:]))
            res = np.empty_like(v)
            res[0] = pp0.sum()
            res[1:] = np.dot(pp0, X)
            res[1:] += (2 * l2) * v[1:]
            return res

    else:
        # dealing with multi-class logistic regression
        beta__ = beta_.reshape((-1, 1 + X.shape[1]))  # (nClasses, nSamples)
        beta_shape = beta__.shape
        # pp - array of class probabilities, shape=(nSamples, nClasses)
        pp = beta__[np.newaxis, :, 0] + np.dot(X, beta__[:, 1:].T)
        pp -= pp.max(axis=1, keepdims=True)
        np.exp(pp, out=pp)
        pp /= pp.sum(axis=1, keepdims=True)

        def hessp(v):
            v2 = v.reshape(beta_shape)
            r_yhat = v2[np.newaxis, :, 0] + np.dot(X, v2[:, 1:].T)
            r_yhat += (-pp * r_yhat).sum(axis=1)[:, np.newaxis]
            r_yhat *= pp
            hessProd = np.zeros(beta_shape)
            hessProd[:, 1:] = np.dot(r_yhat.T, X) + (2 * l2) * v2[:, 1:]
            hessProd[:, 0] = r_yhat.sum(axis=0)
            return hessProd.ravel()

    return gr, hessp
