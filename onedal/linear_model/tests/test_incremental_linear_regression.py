# ==============================================================================
# Copyright 2024 Intel Corporation
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
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from onedal.datatypes import from_table
from onedal.linear_model import IncrementalLinearRegression
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diabetes(queue, dtype):
    X, y = load_diabetes(return_X_y=True)
    X, y = X.astype(dtype), y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=777
    )
    X_train_split = np.array_split(X_train, 2)
    y_train_split = np.array_split(y_train, 2)
    model = IncrementalLinearRegression(fit_intercept=True)
    for i in range(2):
        model.partial_fit(X_train_split[i], y_train_split[i], queue=queue)
    model.finalize_fit()
    y_pred = model.predict(X_test, queue=queue)
    assert mean_squared_error(y_test, y_pred) < 2396


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_blocks", [1, 2, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_results(queue, num_blocks, dtype):
    seed = 42
    num_features, num_targets = 19, 7
    num_samples_train, num_samples_test = 3500, 1999

    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples_train, num_features), dtype=dtype)
    y = X @ coef + intercept[np.newaxis, :]
    X_split = np.array_split(X, num_blocks)
    y_split = np.array_split(y, num_blocks)

    model = IncrementalLinearRegression(fit_intercept=True)
    for i in range(num_blocks):
        model.partial_fit(X_split[i], y_split[i], queue=queue)
    model.finalize_fit()

    if queue and queue.sycl_device.is_gpu:
        tol = 5e-3 if model.coef_.dtype == np.float32 else 1e-5
    else:
        tol = 3e-3 if model.coef_.dtype == np.float32 else 1e-5
    atol = 1e-4 if model.coef_.dtype == np.float32 else 1e-6
    assert_allclose(coef, model.coef_.T, rtol=tol, atol=atol)

    tol = 3e-3 if model.intercept_.dtype == np.float32 else 1e-5
    assert_allclose(intercept, model.intercept_, rtol=tol)

    Xt = gen.random(size=(num_samples_test, num_features), dtype=dtype)
    gtr = Xt @ coef + intercept[np.newaxis, :]

    res = model.predict(Xt, queue=queue)

    tol = 2e-4 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_blocks", [1, 2, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_intercept_results(queue, num_blocks, dtype):
    seed = 42
    num_features, num_targets = 19, 7
    num_samples_train, num_samples_test = 3500, 1999

    gen = np.random.default_rng(seed)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples_train, num_features), dtype=dtype)
    y = X @ coef

    X_split = np.array_split(X, num_blocks)
    y_split = np.array_split(y, num_blocks)

    model = IncrementalLinearRegression(fit_intercept=False)
    for i in range(num_blocks):
        model.partial_fit(X_split[i], y_split[i], queue=queue)
    model.finalize_fit()

    # TODO Find out is it necessary to have accuracy so different for float32 and float64
    if queue and queue.sycl_device.is_gpu:
        tol = 3e-3 if model.coef_.dtype == np.float32 else 1e-7
    else:
        tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-7
    assert_allclose(coef, model.coef_.T, rtol=tol)

    Xt = gen.random(size=(num_samples_test, num_features), dtype=dtype)
    gtr = Xt @ coef

    res = model.predict(Xt, queue=queue)

    tol = 5e-5 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reconstruct_model(queue, dtype):
    seed = 42
    num_samples = 3500
    num_features, num_targets = 14, 9

    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    gtr = X @ coef + intercept[np.newaxis, :]

    model = IncrementalLinearRegression(fit_intercept=True)
    model.coef_ = coef.T
    model.intercept_ = intercept

    res = model.predict(X, queue=queue)

    tol = 1e-5 if res.dtype == np.float32 else 1e-7
    assert_allclose(gtr, res, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incremental_estimator_pickle(queue, dtype):
    import pickle

    from onedal.linear_model import IncrementalLinearRegression

    inclr = IncrementalLinearRegression()

    # Check that estimator can be serialized without any data.
    dump = pickle.dumps(inclr)
    inclr_loaded = pickle.loads(dump)
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(10, 10))
    X = X.astype(dtype)
    coef = gen.random(size=(1, 10), dtype=dtype).T
    y = X @ coef
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)
    inclr.partial_fit(X_split[0], y_split[0], queue=queue)
    inclr_loaded.partial_fit(X_split[0], y_split[0], queue=queue)

    # inclr.finalize_fit()

    assert inclr._need_to_finalize == True
    assert inclr_loaded._need_to_finalize == True

    # Check that estimator can be serialized after partial_fit call.
    dump = pickle.dumps(inclr)
    inclr_loaded = pickle.loads(dump)

    partial_xtx = from_table(inclr._partial_result.partial_xtx)
    partial_xtx_loaded = from_table(inclr_loaded._partial_result.partial_xtx)
    assert_allclose(partial_xtx, partial_xtx_loaded)

    partial_xty = from_table(inclr._partial_result.partial_xty)
    partial_xty_loaded = from_table(inclr_loaded._partial_result.partial_xty)
    assert_allclose(partial_xty, partial_xty_loaded)

    assert inclr._need_to_finalize == False
    # Finalize is called during serialization to make sure partial results are finalized correctly.
    assert inclr_loaded._need_to_finalize == False

    inclr.partial_fit(X_split[1], y_split[1], queue=queue)
    inclr_loaded.partial_fit(X_split[1], y_split[1], queue=queue)
    assert inclr._need_to_finalize == True
    assert inclr_loaded._need_to_finalize == True

    dump = pickle.dumps(inclr_loaded)
    inclr_loaded = pickle.loads(dump)

    assert inclr._need_to_finalize == True
    assert inclr_loaded._need_to_finalize == False

    inclr.finalize_fit()
    inclr_loaded.finalize_fit()

    # Check that finalized estimator can be serialized.
    dump = pickle.dumps(inclr_loaded)
    inclr_loaded = pickle.loads(dump)

    assert_allclose(inclr.coef_, inclr_loaded.coef_, atol=1e-6)
    assert_allclose(inclr.intercept_, inclr_loaded.intercept_, atol=1e-6)
