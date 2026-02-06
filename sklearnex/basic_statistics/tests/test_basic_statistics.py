# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse as sp

from daal4py.sklearn._utils import daal_check_version
from onedal.basic_statistics.tests.utils import options_and_tests
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
    get_queues,
)
from sklearnex import config_context
from sklearnex.basic_statistics import BasicStatistics
from sklearnex.tests.utils import gen_sparse_dataset


# Compute the basic statistics on sparse data on CPU or GPU depending on the queue
def compute_sparse_result(X_sparse, options, queue):
    if queue is not None and queue.sycl_device.is_gpu:
        with config_context(target_offload="gpu"):
            basicstat = BasicStatistics(result_options=options)
            result = basicstat.fit(X_sparse)
    else:
        basicstat = BasicStatistics(result_options=options)
        result = basicstat.fit(X_sparse)
    return result


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_basic_statistics(dataframe, queue):
    X = np.array([[0, 0], [1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    weights = np.array([1, 0.5])
    weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)

    result = BasicStatistics().fit(X_df)

    expected_mean = np.array([0.5, 0.5])
    expected_min = np.array([0, 0])
    expected_max = np.array([1, 1])

    assert_allclose(expected_mean, result.mean_)
    assert_allclose(expected_max, result.max_)
    assert_allclose(expected_min, result.min_)

    result = BasicStatistics().fit(X_df, sample_weight=weights_df)

    expected_weighted_mean = np.array([0.25, 0.25])
    expected_weighted_min = np.array([0, 0])
    expected_weighted_max = np.array([0.5, 0.5])

    assert_allclose(expected_weighted_mean, result.mean_)
    assert_allclose(expected_weighted_min, result.min_)
    assert_allclose(expected_weighted_max, result.max_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_gold_data(dataframe, queue, weighted, dtype):
    X = np.array([[0, 0], [1, 1]])
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = np.array([1, 0.5])
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics()

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    if weighted:
        expected_weighted_mean = np.array([0.25, 0.25])
        expected_weighted_min = np.array([0, 0])
        expected_weighted_max = np.array([0.5, 0.5])
        assert_allclose(expected_weighted_mean, result.mean_)
        assert_allclose(expected_weighted_max, result.max_)
        assert_allclose(expected_weighted_min, result.min_)
    else:
        expected_mean = np.array([0.5, 0.5])
        expected_min = np.array([0, 0])
        expected_max = np.array([1, 1])
        assert_allclose(expected_mean, result.mean_)
        assert_allclose(expected_max, result.max_)
        assert_allclose(expected_min, result.min_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("result_option", options_and_tests.keys())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_option_on_random_data(
    dataframe, queue, result_option, row_count, column_count, weighted, dtype
):
    function, tols = options_and_tests[result_option]
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=result_option)

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res = getattr(result, result_option + "_")
    if weighted:
        weighted_data = np.diag(weights) @ X
        gtr = function(weighted_data)
    else:
        gtr = function(X)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("result_option", options_and_tests.keys())
@pytest.mark.parametrize("row_count", [500, 2000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_option_on_random_sparse_data(
    queue, result_option, row_count, column_count, dtype
):
    if not daal_check_version((2025, "P", 200)) and result_option in [
        "max",
        "sum_squares",
    ]:
        pytest.skip(
            "'max' and 'sum_squares' calculate using a subset of the data in oneDAL version < 2025.2"
        )

    function, tols = options_and_tests[result_option]
    fp32tol, fp64tol = tols
    seed = 77

    gen = np.random.default_rng(seed)

    X_sparse = gen_sparse_dataset(
        row_count,
        column_count,
        density=0.01,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )

    X_dense = X_sparse.toarray()

    result = compute_sparse_result(X_sparse, result_option, queue)

    res = getattr(result, result_option + "_")

    gtr = function(X_dense)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_data(
    dataframe, queue, row_count, column_count, weighted, dtype
):
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=["mean", "max", "sum"])

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res_mean, res_max, res_sum = result.mean_, result.max_, result.sum_
    if weighted:
        weighted_data = np.diag(weights) @ X
        gtr_mean, gtr_max, gtr_sum = (
            options_and_tests["mean"][0](weighted_data),
            options_and_tests["max"][0](weighted_data),
            options_and_tests["sum"][0](weighted_data),
        )
    else:
        gtr_mean, gtr_max, gtr_sum = (
            options_and_tests["mean"][0](X),
            options_and_tests["max"][0](X),
            options_and_tests["sum"][0](X),
        )

    tol = 5e-4 if res_mean.dtype == np.float32 else 1e-7
    assert_allclose(gtr_mean, res_mean, atol=tol)
    assert_allclose(gtr_max, res_max, atol=tol)
    assert_allclose(gtr_sum, res_sum, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_sparse_data(queue, row_count, column_count, dtype):
    seed = 77

    gen = np.random.default_rng(seed)

    X_sparse = gen_sparse_dataset(
        row_count,
        column_count,
        density=0.05,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )

    X_dense = X_sparse.toarray()

    options = [
        "sum",
        "min",
        "mean",
        "standard_deviation",
        "variance",
        "second_order_raw_moment",
    ]

    result = compute_sparse_result(X_sparse, options, queue)

    for result_option in options_and_tests:
        function, tols = options_and_tests[result_option]
        if not result_option in options:
            continue
        fp32tol, fp64tol = tols
        res = getattr(result, result_option + "_")
        gtr = function(X_dense)
        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_option_on_random_data(
    dataframe, queue, row_count, column_count, weighted, dtype
):
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options="all")

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    if weighted:
        weighted_data = np.diag(weights) @ X

    for result_option in options_and_tests:
        function, tols = options_and_tests[result_option]
        fp32tol, fp64tol = tols
        res = getattr(result, result_option + "_")
        if weighted:
            gtr = function(weighted_data)
        else:
            gtr = function(X)
        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_option_on_random_sparse_data(queue, row_count, column_count, dtype):
    seed = 77

    gen = np.random.default_rng(seed)

    X_sparse = gen_sparse_dataset(
        row_count,
        column_count,
        density=0.05,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )
    X_dense = X_sparse.toarray()

    result = compute_sparse_result(X_sparse, "all", queue)

    for result_option in options_and_tests:
        if not daal_check_version((2025, "P", 200)) and result_option in [
            "max",
            "sum_squares",
        ]:
            continue
        function, tols = options_and_tests[result_option]
        fp32tol, fp64tol = tols
        res = getattr(result, result_option + "_")

        gtr = function(X_dense)

        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("result_option", options_and_tests.keys())
@pytest.mark.parametrize("data_size", [100, 1000])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_1d_input_on_random_data(
    dataframe, queue, result_option, data_size, weighted, dtype
):
    function, tols = options_and_tests[result_option]
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=data_size)
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=data_size)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=result_option)

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res = getattr(result, result_option + "_")
    if weighted:
        weighted_data = weights * X
        gtr = function(weighted_data)
    else:
        gtr = function(X)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


def test_warning():
    basicstat = BasicStatistics("all")
    data = np.array([0, 1])

    basicstat.fit(data)
    for i in basicstat._onedal_estimator.get_all_result_options():
        with pytest.warns(
            UserWarning,
            match="Result attributes without a trailing underscore were deprecated in version 2025.1 and will be removed in 2026.0",
        ) as warn_record:
            getattr(basicstat, i)

        if daal_check_version((2026, "P", 0)):
            assert len(warn_record) == 0, i
        else:
            assert len(warn_record) == 1, i
