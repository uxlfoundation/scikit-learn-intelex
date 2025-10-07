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


@pytest.mark.parametrize("nrows", [10, 20])
@pytest.mark.parametrize("ncols", [10, 20])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("l1_ratio", [0.0, 1.0, 0.5])
def test_enet_is_correct(nrows, ncols, fit_intercept, positive, l1_ratio):
    X, y = make_regression(n_samples=nrows, n_features=ncols, random_state=123)
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

    np.testing.assert_allclose(model_d4p.coef_, model_skl.coef_, atol=1e-6, rtol=1e-6)
    if fit_intercept:
        np.testing.assert_allclose(
            model_d4p.intercept_, model_skl.intercept_, atol=1e-6, rtol=1e-6
        )

    if positive:
        assert np.all(model_d4p.coef_ >= 0)


@pytest.mark.parametrize("nrows", [10, 20])
@pytest.mark.parametrize("ncols", [10, 20])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("alpha", [1e-2, 1e2])
def test_lasso_is_correct(nrows, ncols, fit_intercept, positive, alpha):
    X, y = make_regression(n_samples=nrows, n_features=ncols, random_state=123)
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

    tol = 1e-4 if alpha < 1 else 1e-6
    np.testing.assert_allclose(model_d4p.coef_, model_skl.coef_, atol=tol, rtol=tol)
    if fit_intercept:
        np.testing.assert_allclose(
            model_d4p.intercept_, model_skl.intercept_, atol=tol, rtol=tol
        )

    if positive:
        assert np.all(model_d4p.coef_ >= 0)
