# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import lstsq
from sklearn.datasets import make_regression

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.tests.utils import _IS_INTEL


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("overdetermined", [False, True])
@pytest.mark.parametrize("multi_output", [False, True])
def test_sklearnex_import_linear(
    dataframe, queue, dtype, macro_block, overdetermined, multi_output
):
    if (not overdetermined or multi_output) and not daal_check_version((2025, "P", 1)):
        pytest.skip("Functionality introduced in later versions")
    if (
        not overdetermined
        and queue
        and queue.sycl_device.is_gpu
        and not daal_check_version((2025, "P", 200))
    ):
        pytest.skip("Functionality introduced in later versions")

    from sklearnex.linear_model import LinearRegression

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 20) if not overdetermined else (20, 5))
    y = rng.standard_normal(size=(X.shape[0], 3) if multi_output else X.shape[0])

    Xi = np.c_[X, np.ones((X.shape[0], 1))]
    expected_coefs = lstsq(Xi, y)[0]
    expected_intercept = expected_coefs[-1]
    expected_coefs = expected_coefs[: X.shape[1]]
    if multi_output:
        expected_coefs = expected_coefs.T

    linreg = LinearRegression()
    if daal_check_version((2024, "P", 0)) and macro_block is not None:
        hparams = LinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

    X = X.astype(dtype=dtype)
    y = y.astype(dtype=dtype)
    y_list = y.tolist()
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    linreg.fit(X, y)

    assert hasattr(linreg, "_onedal_estimator")
    assert "sklearnex" in linreg.__module__

    rtol = 1e-3 if dtype == np.float32 else 1e-5
    assert_allclose(_as_numpy(linreg.coef_), expected_coefs, rtol=rtol)
    assert_allclose(_as_numpy(linreg.intercept_), expected_intercept, rtol=rtol)

    # check that it also works with lists
    if isinstance(X, np.ndarray):
        linreg_list = LinearRegression().fit(X, y_list)
        assert_allclose(linreg_list.coef_, linreg.coef_)
        assert_allclose(linreg_list.intercept_, linreg.intercept_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_lasso(dataframe, queue):
    from sklearnex.linear_model import Lasso

    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert "daal4py" in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.0])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_elastic(dataframe, queue):
    from sklearnex.linear_model import ElasticNet

    X, y = make_regression(n_features=2, random_state=0)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert "daal4py" in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_reconstruct_model(dataframe, queue, dtype):
    from sklearnex.linear_model import LinearRegression

    seed = 42
    num_samples = 3500
    num_features, num_targets = 14, 9

    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    gtr = X @ coef + intercept[np.newaxis, :]

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    linreg = LinearRegression(fit_intercept=True)
    linreg.coef_ = coef.T
    linreg.intercept_ = intercept

    y_pred = linreg.predict(X)

    tol = 1e-5 if _as_numpy(y_pred).dtype == np.float32 else 1e-7
    assert_allclose(gtr, _as_numpy(y_pred), rtol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("problem_type", ["regular", "overdetermined", "singular"])
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 1)),
    reason="Functionality introduced in the versions >= 2025.0",
)
def test_multioutput_regression(dataframe, queue, dtype, fit_intercept, problem_type):
    if (
        problem_type != "regular"
        and queue
        and queue.sycl_device.is_gpu
        and not daal_check_version((2025, "P", 200))
    ):
        pytest.skip("Functionality introduced in later versions")
    from sklearnex.linear_model import LinearRegression

    gen = np.random.default_rng(seed=123)
    if problem_type == "regular":
        X = gen.standard_normal(size=(20, 5))
    elif problem_type == "singular":
        X = gen.standard_normal(size=(20, 4))
        X[:, 3] = X[:, 2]
    else:
        X = gen.standard_normal(size=(10, 20))
    y = gen.standard_normal(size=(X.shape[0], 3), dtype=dtype)

    X_in = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_in = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    model = LinearRegression(fit_intercept=fit_intercept).fit(X_in, y_in)
    if not fit_intercept:
        A = X.T @ X
        b = X.T @ y
        x = model.coef_.T
    else:
        Xi = np.c_[X, np.ones((X.shape[0], 1))]
        A = Xi.T @ Xi
        b = Xi.T @ y
        x = np.r_[model.coef_.T, model.intercept_.reshape((1, -1))]

    residual = A @ x - b
    assert np.all(np.abs(residual) < 1e-5)

    pred = model.predict(X_in)
    expected_pred = X @ model.coef_.T + model.intercept_.reshape((1, -1))
    tol = 1e-5 if pred.dtype == np.float32 else 1e-7
    assert_allclose(_as_numpy(pred), expected_pred, rtol=tol)

    # check that it also works when 'y' is a list of lists
    if dataframe == "numpy":
        y_lists = y.tolist()
        model_lists = LinearRegression(fit_intercept=fit_intercept).fit(X, y_lists)
        assert_allclose(model.coef_, model_lists.coef_)
        if fit_intercept:
            assert_allclose(model.intercept_, model_lists.intercept_)
