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

from os import environ

from daal4py.sklearn._utils import sklearn_check_version

# sklearn requires manual enabling of Scipy array API support
# if `array-api-compat` package is present in environment
# TODO: create generic approach to handle this for all tests
if sklearn_check_version("1.6"):
    environ["SCIPY_ARRAY_API"] = "1"


import numpy as np
import pytest
from numpy.linalg import slogdet
from numpy.testing import assert_allclose
from scipy.linalg import pinvh
from sklearn.covariance.tests.test_covariance import (
    test_covariance,
    test_EmpiricalCovariance_validates_mahalanobis,
)
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_partial_fit_on_gold_data(dataframe, queue, dtype, assume_centered):
    is_gpu = queue is not None and queue.sycl_device.is_gpu
    if assume_centered and is_gpu and not daal_check_version((2025, "P", 0)):
        pytest.skip(
            "Due to a bug on oneDAL side, means are not set to zero when assume_centered=True"
        )
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    if assume_centered:
        expected_covariance = np.array([[0, 0], [0, 1]])
        expected_means = np.array([0, 0])
    else:
        expected_covariance = np.array([[0, 0], [0, 0]])
        expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    if assume_centered:
        expected_covariance = np.array([[5, 10], [10, 20]])
        expected_means = np.array([0, 0])
    else:
        expected_covariance = np.array([[1, 2], [2, 4]])
        expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(dataframe, queue, batch_size, dtype):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1, 2, 3], [0, -1, -2, -3], [0, 1, 2, 3], [0, 1, 2, 3]])
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    inccov = IncrementalEmpiricalCovariance(batch_size=batch_size)

    result = inccov.fit(X_df)

    expected_covariance = np.array(
        [[0, 0, 0, 0], [0, 0.75, 1.5, 2.25], [0, 1.5, 3, 4.5], [0, 2.25, 4.5, 6.75]]
    )
    expected_means = np.array([0, 0.5, 1, 1.5])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_random_data(
    dataframe, queue, num_batches, row_count, column_count, dtype
):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype)
    X_split = np.array_split(X, num_batches)
    inccov = IncrementalEmpiricalCovariance()

    for i in range(num_batches):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    expected_covariance = np.cov(X.T, bias=1)
    expected_means = np.mean(X, axis=0)

    assert_allclose(expected_covariance, result.covariance_, atol=1e-6)
    assert_allclose(expected_means, result.location_, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_fit_on_random_data(
    dataframe, queue, num_batches, row_count, column_count, dtype, assume_centered
):
    is_gpu = queue is not None and queue.sycl_device.is_gpu
    if assume_centered and is_gpu and not daal_check_version((2025, "P", 0)):
        pytest.skip(
            "Due to a bug on oneDAL side, means are not set to zero when assume_centered=True"
        )
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    batch_size = row_count // num_batches
    inccov = IncrementalEmpiricalCovariance(
        batch_size=batch_size, assume_centered=assume_centered
    )

    result = inccov.fit(X_df)

    if assume_centered:
        expected_covariance = np.dot(X.T, X) / X.shape[0]
        expected_means = np.zeros_like(X[0])
    else:
        expected_covariance = np.cov(X.T, bias=1)
        expected_means = np.mean(X, axis=0)

    assert_allclose(expected_covariance, result.covariance_, atol=1e-6)
    assert_allclose(expected_means, result.location_, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_whitened_toy_score(dataframe, queue):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    # Load a sklearn toy dataset with sufficient data
    X, _ = load_diabetes(return_X_y=True)
    n = X.shape[1]

    # Transform the data into uncorrelated, unity variance components
    X = PCA(whiten=True).fit_transform(X)

    # change dataframe
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    # fit data
    est = IncrementalEmpiricalCovariance()
    est.fit(X_df)
    # location_ attribute approximately zero (10,), covariance_ identity (10,10)

    # The log-likelihood can be calculated simply due to covariance_
    # use of scipy.linalg.pinvh, np.linalg.sloget and np.cov for estimator
    # independence
    expected_result = (
        -(n - slogdet(pinvh(np.cov(X.T, bias=1)))[1] + n * np.log(2 * np.pi)) / 2
    )
    # expected_result = -14.1780602988
    result = _as_numpy(est.score(X_df))
    assert_allclose(expected_result, result, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_incremental_estimatior_pickle(dataframe, queue, dtype):
    import pickle

    from sklearnex.covariance import IncrementalEmpiricalCovariance

    inccov = IncrementalEmpiricalCovariance()

    # Check that estimator can be serialized without any data.
    dump = pickle.dumps(inccov)
    inccov_loaded = pickle.loads(dump)

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(10, 10))
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    X_split_df = _convert_to_dataframe(X_split[0], sycl_queue=queue, target_df=dataframe)
    inccov.partial_fit(X_split_df)
    inccov_loaded.partial_fit(X_split_df)

    # Check that estimator can be serialized after partial_fit call.
    dump = pickle.dumps(inccov_loaded)
    inccov_loaded = pickle.loads(dump)

    assert inccov.batch_size == inccov_loaded.batch_size
    assert inccov.n_features_in_ == inccov_loaded.n_features_in_
    assert inccov.n_samples_seen_ == inccov_loaded.n_samples_seen_
    if hasattr(inccov, "_parameter_constraints"):
        assert inccov._parameter_constraints == inccov_loaded._parameter_constraints
    assert inccov.n_jobs == inccov_loaded.n_jobs

    X_split_df = _convert_to_dataframe(X_split[1], sycl_queue=queue, target_df=dataframe)
    inccov.partial_fit(X_split_df)
    inccov_loaded.partial_fit(X_split_df)
    dump = pickle.dumps(inccov)
    inccov_loaded = pickle.loads(dump)

    assert_allclose(inccov.location_, inccov_loaded.location_, atol=1e-6)
    assert_allclose(inccov.covariance_, inccov_loaded.covariance_, atol=1e-6)

    # Check that finalized estimator can be serialized.
    dump = pickle.dumps(inccov_loaded)
    inccov_loaded = pickle.loads(dump)

    assert_allclose(inccov.location_, inccov_loaded.location_, atol=1e-6)
    assert_allclose(inccov.covariance_, inccov_loaded.covariance_, atol=1e-6)


# Monkeypatch IncrementalEmpiricalCovariance into relevant sklearn.covariance tests
@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize(
    "sklearn_test",
    [
        test_covariance,
        test_EmpiricalCovariance_validates_mahalanobis,
    ],
)
def test_IncrementalEmpiricalCovariance_against_sklearn(monkeypatch, sklearn_test):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    class_name = ".".join([sklearn_test.__module__, "EmpiricalCovariance"])
    monkeypatch.setattr(class_name, IncrementalEmpiricalCovariance)
    sklearn_test()
