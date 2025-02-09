# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.linear_model import IncrementalLinearRegression
from sklearnex.tests.utils import _IS_INTEL


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(dataframe, queue, fit_intercept, macro_block, dtype):
    X = np.array([[1], [2]])
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = np.array([[1], [2]])
    y = y.astype(dtype=dtype)
    y_df = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)
    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    inclin.fit(X_df, y_df)

    y_pred = inclin.predict(X_df)
    np_y_pred = _as_numpy(y_pred)

    tol = 5e-5 if dtype == np.float32 else 1e-7
    assert_allclose(inclin.coef_, [1], atol=tol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, [0], atol=tol)
    assert_allclose(np_y_pred, y, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_gold_data(
    dataframe, queue, fit_intercept, macro_block, dtype
):
    X = np.array([[1], [2], [3], [4]])
    X = X.astype(dtype=dtype)
    y = X + 3
    y = y.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)

    inclin = IncrementalLinearRegression()
    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_pred = inclin.predict(X_df)
    np_y_pred = _as_numpy(y_pred)

    assert inclin.n_features_in_ == 1
    tol = 1e-5 if dtype == np.float32 else 1e-7
    assert_allclose(inclin.coef_, [[1]], atol=tol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, 3, atol=tol)

    assert_allclose(np_y_pred, y, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_multitarget_on_gold_data(
    dataframe, queue, fit_intercept, macro_block, dtype
):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    X = X.astype(dtype=dtype)
    y = np.dot(X, [1, 2]) + 3
    y = y.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)

    inclin = IncrementalLinearRegression()
    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_pred = inclin.predict(X_df)
    np_y_pred = _as_numpy(y_pred)

    assert inclin.n_features_in_ == 2
    tol = 1e-7
    if dtype == np.float32:
        tol = 7e-6 if _IS_INTEL else 2e-5

    assert_allclose(inclin.coef_, [1.0, 2.0], atol=tol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, 3.0, atol=tol)

    assert_allclose(np_y_pred, y, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("num_samples", [100, 1000])
@pytest.mark.parametrize("num_features", [5, 10])
@pytest.mark.parametrize("num_targets", [1, 2])
@pytest.mark.parametrize("num_blocks", [1, 10])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_random_data(
    dataframe,
    queue,
    fit_intercept,
    num_samples,
    num_features,
    num_targets,
    num_blocks,
    macro_block,
    dtype,
):
    seed = 42
    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    if fit_intercept:
        y = X @ coef + intercept[np.newaxis, :]
    else:
        y = X @ coef

    X_split = np.array_split(X, num_blocks)
    y_split = np.array_split(y, num_blocks)

    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)
    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(num_blocks):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    tol = 1e-4 if inclin.coef_.dtype == np.float32 else 1e-7
    assert_allclose(coef.T.squeeze(), inclin.coef_, atol=tol)

    if fit_intercept:
        assert_allclose(intercept, inclin.intercept_, atol=tol)

    X_test = gen.random(size=(num_samples, num_features), dtype=dtype)
    if fit_intercept:
        expected_y_pred = (X_test @ coef + intercept[np.newaxis, :]).squeeze()
    else:
        expected_y_pred = (X_test @ coef).squeeze()

    X_test_df = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    y_pred = inclin.predict(X_test_df)

    assert_allclose(expected_y_pred, _as_numpy(y_pred), atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_incremental_estimatior_pickle(dataframe, queue, fit_intercept, dtype):
    import pickle

    from sklearnex.linear_model import IncrementalLinearRegression

    inclin = IncrementalLinearRegression()

    # Check that estimator can be serialized without any data.
    dump = pickle.dumps(inclin)
    inclin_loaded = pickle.loads(dump)

    seed = 77
    gen = np.random.default_rng(seed)
    intercept = gen.random(size=1, dtype=dtype)
    coef = gen.random(size=(1, 10), dtype=dtype).T
    X = gen.uniform(low=-0.3, high=+0.7, size=(30, 10))
    X = X.astype(dtype)
    if fit_intercept:
        y = X @ coef + intercept[np.newaxis, :]
    else:
        y = X @ coef
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)
    X_split_df = _convert_to_dataframe(X_split[0], sycl_queue=queue, target_df=dataframe)
    y_split_df = _convert_to_dataframe(y_split[0], sycl_queue=queue, target_df=dataframe)
    inclin.partial_fit(X_split_df, y_split_df)
    inclin_loaded.partial_fit(X_split_df, y_split_df)

    # Check that estimator can be serialized after partial_fit call.
    dump = pickle.dumps(inclin_loaded)
    inclin_loaded = pickle.loads(dump)

    assert inclin.batch_size == inclin_loaded.batch_size
    assert inclin.n_features_in_ == inclin_loaded.n_features_in_
    assert inclin.n_samples_seen_ == inclin_loaded.n_samples_seen_
    if hasattr(inclin, "_parameter_constraints"):
        assert inclin._parameter_constraints == inclin_loaded._parameter_constraints
    assert inclin.n_jobs == inclin_loaded.n_jobs

    X_split_df = _convert_to_dataframe(X_split[1], sycl_queue=queue, target_df=dataframe)
    y_split_df = _convert_to_dataframe(y_split[1], sycl_queue=queue, target_df=dataframe)
    inclin.partial_fit(X_split_df, y_split_df)
    inclin_loaded.partial_fit(X_split_df, y_split_df)
    dump = pickle.dumps(inclin)
    inclin_loaded = pickle.loads(dump)

    assert_allclose(inclin.coef_, inclin_loaded.coef_, atol=1e-6)
    assert_allclose(inclin.intercept_, inclin_loaded.intercept_, atol=1e-6)

    # Check that finalized estimator can be serialized.
    dump = pickle.dumps(inclin_loaded)
    inclin_loaded = pickle.loads(dump)

    assert_allclose(inclin.coef_, inclin_loaded.coef_, atol=1e-6)
    assert_allclose(inclin.intercept_, inclin_loaded.intercept_, atol=1e-6)
