# ==============================================================================
# Copyright contributors to the oneDAL project
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
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet as _sklElasticnet
from sklearn.linear_model import Lasso as _sklLasso

from daal4py.sklearn.linear_model import ElasticNet, Lasso


def fn_lasso(model, X, y, lambda_):
    resid = y - model.predict(X)
    fn_ssq = resid.reshape(-1) @ resid.reshape(-1)
    fn_l1 = np.abs(model.coef_).sum()
    return (1 / (2 * X.shape[0])) * fn_ssq + lambda_ * fn_l1


@pytest.mark.parametrize("nrows", [10, 20])
@pytest.mark.parametrize("ncols", [10, 20])
@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("l1_ratio", [0.0, 1.0, 0.5])
def test_enet_is_correct(nrows, ncols, n_targets, fit_intercept, positive, l1_ratio):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_targets=n_targets, random_state=123
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model_d4p = ElasticNet(
            fit_intercept=fit_intercept,
            positive=positive,
            l1_ratio=l1_ratio,
            tol=1e-7,
            max_iter=int(1e4),
        ).fit(X, y)
        model_skl = _sklElasticnet(
            fit_intercept=fit_intercept,
            positive=positive,
            l1_ratio=l1_ratio,
            tol=1e-7,
            max_iter=int(1e4),
        ).fit(X, y)

    # Note: lasso is not guaranteed to have a unique global optimum.
    # If the coefficients do not match, this makes another check on
    # the optimality of the function values instead. It checks that
    # the result from daal4py is no worse than scikit-learn's.

    tol = 1e-6 if n_targets == 1 else 1e-5
    try:
        np.testing.assert_allclose(model_d4p.coef_, model_skl.coef_, atol=tol, rtol=tol)
    except AssertionError as e:
        if l1_ratio != 1:
            raise e
        fn_d4p = fn_lasso(model_d4p, X, y, model_d4p.alpha)
        fn_skl = fn_lasso(model_skl, X, y, model_skl.alpha)
        assert fn_d4p <= fn_skl

    if fit_intercept:
        np.testing.assert_allclose(
            model_d4p.intercept_, model_skl.intercept_, atol=tol, rtol=tol
        )

    if positive:
        assert np.all(model_d4p.coef_ >= 0)


@pytest.mark.parametrize("nrows", [10, 20])
@pytest.mark.parametrize("ncols", [10, 20])
@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("alpha", [1e-2, 1e2])
def test_lasso_is_correct(nrows, ncols, n_targets, fit_intercept, positive, alpha):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_targets=n_targets, random_state=123
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model_d4p = Lasso(
            fit_intercept=fit_intercept,
            positive=positive,
            alpha=alpha,
            tol=1e-7,
            max_iter=int(1e4),
        ).fit(X, y)
        model_skl = _sklLasso(
            fit_intercept=fit_intercept,
            positive=positive,
            alpha=alpha,
            tol=1e-7,
            max_iter=int(1e4),
        ).fit(X, y)

    tol = 1e-4 if alpha < 1 else (1e-6 if n_targets == 1 else 1e-5)
    try:
        np.testing.assert_allclose(model_d4p.coef_, model_skl.coef_, atol=tol, rtol=tol)
        if fit_intercept:
            np.testing.assert_allclose(
                model_d4p.intercept_, model_skl.intercept_, atol=tol, rtol=tol
            )
    except AssertionError as e:
        fn_d4p = fn_lasso(model_d4p, X, y, model_d4p.alpha)
        fn_skl = fn_lasso(model_skl, X, y, model_skl.alpha)
        assert fn_d4p <= fn_skl

    if positive:
        assert np.all(model_d4p.coef_ >= 0)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_warm_start(n_targets):
    X, y = make_regression(
        n_samples=20, n_features=10, n_targets=n_targets, random_state=123
    )
    X1 = X[:10]
    y1 = y[:10]
    X2 = X[10:]
    y2 = y[10:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model_d4p = ElasticNet(
            warm_start=True,
            tol=1e-7,
            max_iter=int(1e4),
        ).fit(X1, y1)
        coefs_ref = model_d4p.coef_.copy()
        intercept_ref = model_d4p.intercept_.copy()

        model_d4p.set_params(max_iter=1)
        model_d4p.fit(X2, y2)

        model_from_scratch = ElasticNet(tol=1e-7, max_iter=int(1e4)).fit(X2, y2)

    diff_ref = np.linalg.norm(model_d4p.coef_ - coefs_ref)
    diff_from_scratch = np.linalg.norm(model_d4p.coef_ - model_from_scratch.coef_)

    assert diff_ref < diff_from_scratch
