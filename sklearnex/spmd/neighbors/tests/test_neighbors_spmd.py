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

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils.spmd import (
    _assert_unordered_allclose,
    _generate_classification_data,
    _generate_regression_data,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
    _spmd_assert_allclose,
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.mpi
def test_knncls_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
    from sklearnex.spmd.neighbors import KNeighborsClassifier as KNeighborsClassifier_SPMD

    # Create gold data and convert to dataframe
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.9, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    # TODO: handle situations where not all classes are present on all ranks?
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    X_test = np.array(
        [
            [1.0, -0.5],
            [-5.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Ensure predictions of batch algo match spmd
    spmd_model = KNeighborsClassifier_SPMD(n_neighbors=1, algorithm="brute").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = KNeighborsClassifier_Batch(n_neighbors=1, algorithm="brute").fit(
        X_train, y_train
    )
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _assert_unordered_allclose(spmd_indcs, batch_indcs, localize=True)
    _assert_unordered_allclose(spmd_dists, batch_dists, localize=True)
    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features_and_classes", [(5, 2), (25, 2), (25, 10)])
@pytest.mark.parametrize("n_neighbors", [1, 5, 20])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_knncls_spmd_synthetic(
    n_samples,
    n_features_and_classes,
    n_neighbors,
    weights,
    dataframe,
    queue,
    dtype,
    use_raw_input,
    metric="euclidean",
):
    n_features, n_classes = n_features_and_classes
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
    from sklearnex.spmd.neighbors import KNeighborsClassifier as KNeighborsClassifier_SPMD

    # Generate data and convert to dataframe
    X_train, X_test, y_train, _ = _generate_classification_data(
        n_samples, n_features, n_classes, dtype=dtype
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Ensure predictions of batch algo match spmd
    spmd_model = KNeighborsClassifier_SPMD(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    )
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_model.fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = KNeighborsClassifier_Batch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(X_train, y_train)
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    tol = 1e-4
    if dtype == np.float64:
        _assert_unordered_allclose(spmd_indcs, batch_indcs, localize=True)
        _assert_unordered_allclose(
            spmd_dists, batch_dists, localize=True, rtol=tol, atol=tol
        )
    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.mpi
def test_knnreg_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch
    from sklearnex.spmd.neighbors import KNeighborsRegressor as KNeighborsRegressor_SPMD

    # Create gold data and convert to dataframe
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([3.0, 5.0, 4.0, 7.0, 5.0, 6.0, 1.0, 2.0, 0.0])
    X_test = np.array(
        [
            [1.0, -0.5],
            [-5.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Ensure predictions of batch algo match spmd
    spmd_model = KNeighborsRegressor_SPMD(n_neighbors=1, algorithm="brute").fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = KNeighborsRegressor_Batch(n_neighbors=1, algorithm="brute").fit(
        X_train, y_train
    )
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _assert_unordered_allclose(spmd_indcs, batch_indcs, localize=True)
    _assert_unordered_allclose(spmd_dists, batch_dists, localize=True)
    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_neighbors", [1, 5, 20])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "minkowski", "chebyshev", "cosine"]
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_knnreg_spmd_synthetic(
    n_samples,
    n_features,
    n_neighbors,
    weights,
    metric,
    dataframe,
    queue,
    dtype,
    use_raw_input,
):
    # Import spmd and batch algo
    from sklearnex.neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch
    from sklearnex.spmd.neighbors import KNeighborsRegressor as KNeighborsRegressor_SPMD

    # Generate data and convert to dataframe
    X_train, X_test, y_train, _ = _generate_regression_data(
        n_samples, n_features, dtype=dtype
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_y_train = _convert_to_dataframe(
        _get_local_tensor(y_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Ensure predictions of batch algo match spmd
    spmd_model = KNeighborsRegressor_SPMD(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    )
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_model.fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = KNeighborsRegressor_Batch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute"
    ).fit(X_train, y_train)
    spmd_dists, spmd_indcs = spmd_model.kneighbors(local_dpt_X_test)
    batch_dists, batch_indcs = batch_model.kneighbors(X_test)
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    tol = 0.005 if dtype == np.float32 else 1e-4
    if dtype == np.float64:
        _assert_unordered_allclose(spmd_indcs, batch_indcs, localize=True)
        _assert_unordered_allclose(
            spmd_dists, batch_dists, localize=True, rtol=tol, atol=tol
        )
    _spmd_assert_allclose(spmd_result, batch_result, rtol=tol, atol=tol)
