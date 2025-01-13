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
)
from sklearnex.basic_statistics import BasicStatistics

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

    assert_allclose(expected_mean, result.mean)
    assert_allclose(expected_max, result.max)
    assert_allclose(expected_min, result.min)

    result = BasicStatistics().fit(X_df, sample_weight=weights_df)

    expected_weighted_mean = np.array([0.25, 0.25])
    expected_weighted_min = np.array([0, 0])
    expected_weighted_max = np.array([0.5, 0.5])

    assert_allclose(expected_weighted_mean, result.mean)
    assert_allclose(expected_weighted_min, result.min)
    assert_allclose(expected_weighted_max, result.max)


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
        assert_allclose(expected_weighted_mean, result.mean)
        assert_allclose(expected_weighted_max, result.max)
        assert_allclose(expected_weighted_min, result.min)
    else:
        expected_mean = np.array([0.5, 0.5])
        expected_min = np.array([0, 0])
        expected_max = np.array([1, 1])
        assert_allclose(expected_mean, result.mean)
        assert_allclose(expected_max, result.max)
        assert_allclose(expected_min, result.min)


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

    res = getattr(result, result_option)
    if weighted:
        weighted_data = np.diag(weights) @ X
        gtr = function(weighted_data)
    else:
        gtr = function(X)

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

    res_mean, res_max, res_sum = result.mean, result.max, result.sum
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


# @pytest.mark.skipif(not hasattr(sp, "random_array"), reason="requires scipy>=1.12.0")
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_sparse_data(
    dataframe, queue, row_count, column_count, dtype
):
    seed = 77
    random_state = 42

    gen = np.random.default_rng(seed)

    X_sparse = sp.random_array(
        shape=(row_count, column_count),
        density=0.05,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )
    X_dense = X_sparse.toarray()

    options = [
        "sum",
        # TODO: There is a bug in oneDAL's max computations on GPU
        # "max",
        "min",
        "mean",
        "standard_deviation",
        "variance",
        "sum_squares",
        "sum_squares_centered",
        "second_order_raw_moment",
    ]
    basicstat = BasicStatistics(result_options=options)

    result = basicstat.fit(X_sparse)

    gtr_sum, gtr_min, gtr_mean = (
        options_and_tests["sum"][0](X_dense),
        options_and_tests["min"][0](X_dense),
        options_and_tests["mean"][0](X_dense),
    )

    tol = 5e-4 if result.mean_.dtype == np.float32 else 1e-7
    assert_allclose(gtr_sum, result.sum_, atol=tol)
    assert_allclose(gtr_min, result.min_, atol=tol)
    assert_allclose(gtr_mean, result.mean_, atol=tol)
    #assert_allclose(gtr_std, result.standard_deviation_, atol=tol)
    #assert_allclose(gtr_var, result.variance_, atol=tol)
    #assert_allclose(gtr_variation, result.variation_, atol=tol)
    #assert_allclose(gtr_ss, result.sum_squares_, atol=tol)
    #assert_allclose(gtr_ssc, result.sum_squares_centered_, atol=tol)
    #assert_allclose(gtr_seconf_moment, result.second_order_raw_moment_, atol=tol)


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
        res = getattr(result, result_option)
        if weighted:
            gtr = function(weighted_data)
        else:
            gtr = function(X)
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

    res = getattr(result, result_option)
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
