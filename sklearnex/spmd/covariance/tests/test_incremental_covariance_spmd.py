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
    _generate_statistic_data,
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
@pytest.mark.parametrize("assume_centered", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_covariance_fit_spmd_gold(dataframe, queue, assume_centered, dtype):
    # Import spmd and batch algo
    from sklearnex.covariance import IncrementalEmpiricalCovariance
    from sklearnex.spmd.covariance import (
        IncrementalEmpiricalCovariance as IncrementalEmpiricalCovariance_SPMD,
    )

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
            [0.0, 7.0, 128.0],
        ],
        dtype=dtype,
    )

    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # ensure results of batch algo match spmd
    spmd_result = IncrementalEmpiricalCovariance_SPMD(
        assume_centered=assume_centered
    ).fit(local_dpt_data)
    non_spmd_result = IncrementalEmpiricalCovariance(assume_centered=assume_centered).fit(
        dpt_data
    )

    assert_allclose(spmd_result.covariance_, non_spmd_result.covariance_)
    assert_allclose(spmd_result.location_, non_spmd_result.location_)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("assume_centered", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_incremental_covariance_partial_fit_spmd_gold(
    dataframe, queue, num_blocks, assume_centered, dtype
):
    # Import spmd and batch algo
    from sklearnex.covariance import IncrementalEmpiricalCovariance
    from sklearnex.spmd.covariance import (
        IncrementalEmpiricalCovariance as IncrementalEmpiricalCovariance_SPMD,
    )

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
            [0.0, 7.0, 128.0],
        ],
        dtype=dtype,
    )

    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)

    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)

    inccov_spmd = IncrementalEmpiricalCovariance_SPMD(assume_centered=assume_centered)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        inccov_spmd.partial_fit(local_dpt_data)

    inccov.fit(dpt_data)

    assert_allclose(inccov_spmd.covariance_, inccov.covariance_)
    assert_allclose(inccov_spmd.location_, inccov.location_)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("assume_centered", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_incremental_covariance_partial_fit_spmd_synthetic(
    n_samples,
    n_features,
    num_blocks,
    assume_centered,
    dataframe,
    queue,
    dtype,
    use_raw_input,
):
    # Import spmd and batch algo
    from sklearnex.covariance import IncrementalEmpiricalCovariance
    from sklearnex.spmd.covariance import (
        IncrementalEmpiricalCovariance as IncrementalEmpiricalCovariance_SPMD,
    )

    # Generate data and process into dpt
    data = _generate_statistic_data(n_samples, n_features, dtype=dtype)

    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)

    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)

    inccov_spmd = IncrementalEmpiricalCovariance_SPMD(assume_centered=assume_centered)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        # Configure raw input status for spmd estimator
        with config_context(use_raw_input=use_raw_input):
            inccov_spmd.partial_fit(local_dpt_data)

    inccov.fit(dpt_data)

    tol = 1e-7

    assert_allclose(inccov_spmd.covariance_, inccov.covariance_, atol=tol)
    assert_allclose(inccov_spmd.location_, inccov.location_, atol=tol)
