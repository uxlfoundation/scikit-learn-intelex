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
from sklearn.datasets import make_regression

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils.spmd import (
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
def test_rfcls_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestClassifier as RandomForestClassifier_SPMD,
    )

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
    y_train = np.array([0, 2, 1, 2, 1, 0, 1, 2, 0])
    X_test = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
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
    spmd_model = RandomForestClassifier_SPMD(n_estimators=3, random_state=0).fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = RandomForestClassifier_Batch(n_estimators=3, random_state=0).fit(
        X_train, y_train
    )
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 1000])
@pytest.mark.parametrize("n_features_and_classes", [(5, 2), (25, 2), (25, 10)])
@pytest.mark.parametrize("n_estimators", [10, 100])
@pytest.mark.parametrize("max_depth", [3, None])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_rfcls_spmd_synthetic(
    n_samples,
    n_features_and_classes,
    n_estimators,
    max_depth,
    dataframe,
    queue,
    dtype,
    use_raw_input,
):
    n_features, n_classes = n_features_and_classes
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestClassifier as RandomForestClassifier_SPMD,
    )

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
    spmd_model = RandomForestClassifier_SPMD(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    )
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_model.fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = RandomForestClassifier_Batch(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(X_train, y_train)
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
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
def test_rfreg_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestRegressor as RandomForestRegressor_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestRegressor as RandomForestRegressor_SPMD,
    )

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
            [1.0, -1.0],
            [-1.0, 1.0],
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
    spmd_model = RandomForestRegressor_SPMD(n_estimators=3, random_state=0).fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = RandomForestRegressor_Batch(n_estimators=3, random_state=0).fit(
        X_train, y_train
    )
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 1000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_estimators", [10, 100])
@pytest.mark.parametrize("max_depth", [3, None])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_rfreg_spmd_synthetic(
    n_samples, n_features, n_estimators, max_depth, dataframe, queue, dtype, use_raw_input
):
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestRegressor as RandomForestRegressor_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestRegressor as RandomForestRegressor_SPMD,
    )

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
    with config_context(use_raw_input=use_raw_input):
        spmd_model = RandomForestRegressor_Batch(
            n_estimators=n_estimators, max_depth=max_depth, random_state=0
        ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = RandomForestRegressor_Batch(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(X_train, y_train)
    with config_context(use_raw_input=use_raw_input):
        spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    # TODO: remove skips when SPMD and batch are aligned
    pytest.skip("SPMD and batch random forest results not aligned")
    _spmd_assert_allclose(spmd_result, batch_result)
