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
from sklearn.datasets import make_regression

from sklearnex.tests._utils_spmd import (
    _generate_regression_data,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
    _spmd_assert_allclose,
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_linear_spmd_gold():
    # Import spmd and batch algo
    from sklearnex.linear_model import LinearRegression as LinearRegression_Batch
    from sklearnex.spmd.linear_model import LinearRegression as LinearRegression_SPMD

    # Create gold data and process into dpt
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

    local_dpt_X_train = _get_local_tensor(X_train)
    local_dpt_y_train = _get_local_tensor(y_train)
    local_dpt_X_test = _get_local_tensor(X_test)

    # ensure trained model of batch algo matches spmd
    spmd_model = LinearRegression_SPMD().fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = LinearRegression_Batch().fit(X_train, y_train)

    assert_allclose(spmd_model.coef_, batch_model.coef_)
    assert_allclose(spmd_model.intercept_, batch_model.intercept_)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _spmd_assert_allclose(spmd_result, batch_result)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.mpi
def test_linear_spmd_synthetic(n_samples, n_features):
    # Import spmd and batch algo
    from sklearnex.linear_model import LinearRegression as LinearRegression_Batch
    from sklearnex.spmd.linear_model import LinearRegression as LinearRegression_SPMD

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = _generate_regression_data(n_samples, n_features)

    local_dpt_X_train = _get_local_tensor(X_train)
    local_dpt_y_train = _get_local_tensor(y_train)
    local_dpt_X_test = _get_local_tensor(X_test)

    # TODO: support linear regression on wide datasets and remove this skip
    if local_dpt_X_train.shape[0] < n_features:
        pytest.skip(
            "SPMD Linear Regression does not support cases where n_rows_rank < n_features"
        )

    # ensure trained model of batch algo matches spmd
    spmd_model = LinearRegression_SPMD().fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = LinearRegression_Batch().fit(X_train, y_train)

    assert_allclose(spmd_model.coef_, batch_model.coef_, rtol=1e-7, atol=1e-7)
    assert_allclose(spmd_model.intercept_, batch_model.intercept_, rtol=1e-7, atol=1e-7)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _spmd_assert_allclose(spmd_result, batch_result)
