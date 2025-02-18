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
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils.spmd import (
    _generate_regression_data,
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_linear_regression_fit_spmd_gold(
    dataframe, queue, fit_intercept, macro_block, dtype
):
    # Import spmd and non-SPMD algo
    from sklearnex.linear_model import IncrementalLinearRegression
    from sklearnex.spmd.linear_model import (
        IncrementalLinearRegression as IncrementalLinearRegression_SPMD,
    )

    # Create gold data and process into dpt
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 16.0],
            [5.0, 32.0],
            [6.0, 64.0],
            [7.0, 128.0],
            [8.0, 0.0],
            [9.0, 2.0],
            [10.0, 4.0],
            [11.0, 8.0],
            [12.0, 16.0],
            [13.0, 32.0],
            [14.0, 64.0],
            [15.0, 128.0],
        ],
        dtype=dtype,
    )
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    local_dpt_X = _convert_to_dataframe(local_X, sycl_queue=queue, target_df=dataframe)

    y = np.dot(X, [1, 2]) + 3
    dpt_y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    local_y = _get_local_tensor(y)
    local_dpt_y = _convert_to_dataframe(local_y, sycl_queue=queue, target_df=dataframe)

    inclin_spmd = IncrementalLinearRegression_SPMD(fit_intercept=fit_intercept)
    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)

    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

        hparams_spmd = IncrementalLinearRegression_SPMD.get_hyperparameters("fit")
        hparams_spmd.cpu_macro_block = macro_block
        hparams_spmd.gpu_macro_block = macro_block

    inclin_spmd.fit(local_dpt_X, local_dpt_y)
    inclin.fit(dpt_X, dpt_y)

    rtol = 1e-5 if (dtype == np.float32) else 1e-7
    assert_allclose(inclin.coef_, inclin_spmd.coef_, rtol=rtol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, inclin_spmd.intercept_, rtol=rtol)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_incremental_linear_regression_partial_fit_spmd_gold(
    dataframe, queue, fit_intercept, num_blocks, macro_block, dtype, use_raw_input
):
    # Import spmd and non-SPMD algo
    from sklearnex.linear_model import IncrementalLinearRegression
    from sklearnex.spmd.linear_model import (
        IncrementalLinearRegression as IncrementalLinearRegression_SPMD,
    )

    # Create gold data and process into dpt
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 16.0],
            [5.0, 32.0],
            [6.0, 64.0],
            [7.0, 128.0],
            [8.0, 0.0],
            [9.0, 2.0],
            [10.0, 4.0],
            [11.0, 8.0],
            [12.0, 16.0],
            [13.0, 32.0],
            [14.0, 64.0],
            [15.0, 128.0],
        ],
        dtype=dtype,
    )
    dpt_X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X)
    split_local_X = np.array_split(local_X, num_blocks)

    y = np.dot(X, [1, 2]) + 3
    dpt_y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    local_y = _get_local_tensor(y)
    split_local_y = np.array_split(local_y, num_blocks)

    inclin_spmd = IncrementalLinearRegression_SPMD(fit_intercept=fit_intercept)
    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)

    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

        hparams_spmd = IncrementalLinearRegression_SPMD.get_hyperparameters("fit")
        hparams_spmd.cpu_macro_block = macro_block
        hparams_spmd.gpu_macro_block = macro_block

    for i in range(num_blocks):
        local_dpt_X = _convert_to_dataframe(
            split_local_X[i], sycl_queue=queue, target_df=dataframe
        )
        local_dpt_y = _convert_to_dataframe(
            split_local_y[i], sycl_queue=queue, target_df=dataframe
        )
        # Configure raw input status for spmd estimator
        with config_context(use_raw_input=use_raw_input):
            inclin_spmd.partial_fit(local_dpt_X, local_dpt_y)

    inclin.fit(dpt_X, dpt_y)

    rtol = 1e-5 if (dtype == np.float32) else 1e-7
    assert_allclose(inclin.coef_, inclin_spmd.coef_, rtol=rtol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, inclin_spmd.intercept_, rtol=rtol)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("num_samples", [100, 1000])
@pytest.mark.parametrize("num_features", [5, 10])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_linear_regression_fit_spmd_random(
    dataframe, queue, fit_intercept, num_samples, num_features, macro_block, dtype
):
    # Import spmd and non-SPMD algo
    from sklearnex.linear_model import IncrementalLinearRegression
    from sklearnex.spmd.linear_model import (
        IncrementalLinearRegression as IncrementalLinearRegression_SPMD,
    )

    tol = 5e-3 if dtype == np.float32 else 1e-7

    # Generate random data and process into dpt
    X_train, X_test, y_train, _ = _generate_regression_data(
        num_samples, num_features, dtype
    )
    dpt_X = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X_train)
    local_dpt_X = _convert_to_dataframe(local_X, sycl_queue=queue, target_df=dataframe)

    dpt_y = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    local_y = _get_local_tensor(y_train)
    local_dpt_y = _convert_to_dataframe(local_y, sycl_queue=queue, target_df=dataframe)

    inclin_spmd = IncrementalLinearRegression_SPMD(fit_intercept=fit_intercept)
    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)

    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

        hparams_spmd = IncrementalLinearRegression_SPMD.get_hyperparameters("fit")
        hparams_spmd.cpu_macro_block = macro_block
        hparams_spmd.gpu_macro_block = macro_block

    inclin_spmd.fit(local_dpt_X, local_dpt_y)
    inclin.fit(dpt_X, dpt_y)

    assert_allclose(inclin.coef_, inclin_spmd.coef_, atol=tol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, inclin_spmd.intercept_, atol=tol)

    y_pred_spmd = inclin_spmd.predict(dpt_X_test)
    y_pred = inclin.predict(dpt_X_test)

    assert_allclose(_as_numpy(y_pred_spmd), _as_numpy(y_pred), atol=tol)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("num_samples", [100, 1000])
@pytest.mark.parametrize("num_features", [5, 10])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_linear_regression_partial_fit_spmd_random(
    dataframe,
    queue,
    fit_intercept,
    num_blocks,
    num_samples,
    num_features,
    macro_block,
    dtype,
):
    # Import spmd and non-SPMD algo
    from sklearnex.linear_model import IncrementalLinearRegression
    from sklearnex.spmd.linear_model import (
        IncrementalLinearRegression as IncrementalLinearRegression_SPMD,
    )

    tol = 5e-3 if dtype == np.float32 else 1e-7

    # Generate random data and process into dpt
    X_train, X_test, y_train, _ = _generate_regression_data(
        num_samples, num_features, dtype, 573
    )
    dpt_X = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    dpt_X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    local_X = _get_local_tensor(X_train)
    X_split = np.array_split(X_train, num_blocks)
    split_local_X = np.array_split(local_X, num_blocks)

    dpt_y = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    y_split = np.array_split(y_train, num_blocks)
    local_y = _get_local_tensor(y_train)
    split_local_y = np.array_split(local_y, num_blocks)

    inclin_spmd = IncrementalLinearRegression_SPMD(fit_intercept=fit_intercept)
    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)

    if macro_block is not None:
        hparams = IncrementalLinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block

        hparams_spmd = IncrementalLinearRegression_SPMD.get_hyperparameters("fit")
        hparams_spmd.cpu_macro_block = macro_block
        hparams_spmd.gpu_macro_block = macro_block

    for i in range(num_blocks):
        local_dpt_X = _convert_to_dataframe(
            split_local_X[i], sycl_queue=queue, target_df=dataframe
        )
        local_dpt_y = _convert_to_dataframe(
            split_local_y[i], sycl_queue=queue, target_df=dataframe
        )
        dpt_X = _convert_to_dataframe(X_split[i], sycl_queue=queue, target_df=dataframe)
        dpt_y = _convert_to_dataframe(y_split[i], sycl_queue=queue, target_df=dataframe)

        inclin_spmd.partial_fit(local_dpt_X, local_dpt_y)
        inclin.partial_fit(dpt_X, dpt_y)

    assert_allclose(inclin.coef_, inclin_spmd.coef_, atol=tol)
    if fit_intercept:
        assert_allclose(inclin.intercept_, inclin_spmd.intercept_, atol=tol)

    y_pred_spmd = inclin_spmd.predict(dpt_X_test)
    y_pred = inclin.predict(dpt_X_test)

    assert_allclose(_as_numpy(y_pred_spmd), _as_numpy(y_pred), atol=tol)
