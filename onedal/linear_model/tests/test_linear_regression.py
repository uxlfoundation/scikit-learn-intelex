# ===============================================================================
# Copyright 2023 Intel Corporation
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
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from daal4py.sklearn._utils import daal_check_version
from onedal.linear_model import LinearRegression
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diabetes(queue, dtype):
    X, y = load_diabetes(return_X_y=True)
    X, y = X.astype(dtype), y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=777
    )
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train, queue=queue)
    y_pred = model.predict(X_test, queue=queue)
    assert_allclose(mean_squared_error(y_test, y_pred), 2395.567, rtol=1e-5)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pickle(queue, dtype):
    X, y = load_diabetes(return_X_y=True)
    X, y = X.astype(dtype), y.astype(dtype)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y, queue=queue)
    expected = model.predict(X, queue=queue)

    import pickle

    dump = pickle.dumps(model)
    model2 = pickle.loads(dump)

    assert isinstance(model2, model.__class__)
    result = model2.predict(X, queue=queue)

    assert_array_equal(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_results(queue, dtype):
    seed = 42
    f_count, r_count = 19, 7
    s_count, t_count = 3500, 1999

    gen = np.random.default_rng(seed)
    intp = gen.random(size=r_count, dtype=dtype)
    coef = gen.random(size=(r_count, f_count), dtype=dtype).T

    X = gen.random(size=(s_count, f_count), dtype=dtype)
    y = X @ coef + intp[np.newaxis, :]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y, queue=queue)

    if queue and queue.sycl_device.is_gpu:
        tol = 5e-3 if model.coef_.dtype == np.float32 else 1e-5
    else:
        tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-5
    assert_allclose(coef, model.coef_.T, rtol=tol)

    tol = 2e-3 if model.intercept_.dtype == np.float32 else 1e-5
    assert_allclose(intp, model.intercept_, rtol=tol)

    Xt = gen.random(size=(t_count, f_count), dtype=dtype)
    gtr = Xt @ coef + intp[np.newaxis, :]

    res = model.predict(Xt, queue=queue)

    tol = 2e-4 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_intercept_results(queue, dtype):
    seed = 42
    f_count, r_count = 19, 7
    s_count, t_count = 3500, 1999

    gen = np.random.default_rng(seed)
    coef = gen.random(size=(r_count, f_count), dtype=dtype).T

    X = gen.random(size=(s_count, f_count), dtype=dtype)
    y = X @ coef

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y, queue=queue)

    if queue and queue.sycl_device.is_gpu:
        tol = 3e-3 if model.coef_.dtype == np.float32 else 1e-7
    else:
        tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-7
    assert_allclose(coef, model.coef_.T, rtol=tol)

    Xt = gen.random(size=(t_count, f_count), dtype=dtype)
    gtr = Xt @ coef

    res = model.predict(Xt, queue=queue)

    tol = 5e-5 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reconstruct_model(queue, dtype):
    seed = 42
    s_count = 3500
    f_count, r_count = 14, 9

    gen = np.random.default_rng(seed)
    intp = gen.random(size=r_count, dtype=dtype)
    coef = gen.random(size=(r_count, f_count), dtype=dtype).T

    X = gen.random(size=(s_count, f_count), dtype=dtype)
    gtr = X @ coef + intp[np.newaxis, :]

    model = LinearRegression(fit_intercept=True)
    model.coef_ = coef.T
    model.intercept_ = intp

    res = model.predict(X, queue=queue)

    tol = 1e-5 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 1)),
    reason="Functionality introduced in later versions",
)
def test_overdetermined_system(queue, dtype, fit_intercept):
    if queue and queue.sycl_device.is_gpu and not daal_check_version((2025, "P", 200)):
        pytest.skip("Functionality introduced in later versions")
    gen = np.random.default_rng(seed=123)
    X = gen.standard_normal(size=(10, 20))
    y = gen.standard_normal(size=X.shape[0])

    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    if not fit_intercept:
        A = X.T @ X
        b = X.T @ y
        x = model.coef_
    else:
        Xi = np.c_[X, np.ones((X.shape[0], 1))]
        A = Xi.T @ Xi
        b = Xi.T @ y
        x = np.r_[model.coef_, model.intercept_]
    residual = A @ x - b
    assert np.all(np.abs(residual) < 1e-6)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 1)),
    reason="Functionality introduced in later versions",
)
def test_singular_matrix(queue, dtype, fit_intercept):
    if queue and queue.sycl_device.is_gpu and not daal_check_version((2025, "P", 200)):
        pytest.skip("Functionality introduced in later versions")
    gen = np.random.default_rng(seed=123)
    X = gen.standard_normal(size=(20, 4))
    X[:, 2] = X[:, 3]
    y = gen.standard_normal(size=X.shape[0])

    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    if not fit_intercept:
        A = X.T @ X
        b = X.T @ y
        x = model.coef_
    else:
        Xi = np.c_[X, np.ones((X.shape[0], 1))]
        A = Xi.T @ Xi
        b = Xi.T @ y
        x = np.r_[model.coef_, model.intercept_]
    residual = A @ x - b
    assert np.all(np.abs(residual) < 1e-6)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("problem_type", ["regular", "overdetermined", "singular"])
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 1)),
    reason="Functionality introduced in the versions >= 2025.0",
)
def test_multioutput_regression(queue, dtype, fit_intercept, problem_type):
    if (
        problem_type != "regular"
        and queue
        and queue.sycl_device.is_gpu
        and not daal_check_version((2025, "P", 200))
    ):
        pytest.skip("Functionality introduced in later versions")
    gen = np.random.default_rng(seed=123)
    if problem_type == "regular":
        X = gen.standard_normal(size=(20, 5))
    elif problem_type == "singular":
        X = gen.standard_normal(size=(20, 4))
        X[:, 3] = X[:, 2]
    else:
        X = gen.standard_normal(size=(10, 20))
    Y = gen.standard_normal(size=(X.shape[0], 3), dtype=dtype)

    model = LinearRegression(fit_intercept=fit_intercept).fit(X, Y)
    if not fit_intercept:
        A = X.T @ X
        b = X.T @ Y
        x = model.coef_.T
    else:
        Xi = np.c_[X, np.ones((X.shape[0], 1))]
        A = Xi.T @ Xi
        b = Xi.T @ Y
        x = np.r_[model.coef_.T, model.intercept_.reshape((1, -1))]
    residual = A @ x - b
    assert np.all(np.abs(residual) < 1e-5)

    pred = model.predict(X, queue=queue)
    expected_pred = X @ model.coef_.T + model.intercept_.reshape((1, -1))
    tol = 1e-5 if pred.dtype == np.float32 else 1e-7
    assert_allclose(pred, expected_pred, rtol=tol)

    # check that it also works when 'y' is a list of lists
    Y_lists = Y.tolist()
    model_lists = LinearRegression(fit_intercept=fit_intercept).fit(X, Y_lists)
    assert_allclose(model.coef_, model_lists.coef_)
    if fit_intercept:
        assert_allclose(model.intercept_, model_lists.intercept_)
