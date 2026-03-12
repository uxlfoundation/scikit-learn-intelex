# ==============================================================================
# Copyright Contributors to the oneDAL Project
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
from sklearnex.tests.utils.spmd import _get_local_tensor, _mpi_libs_and_gpu_available


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_max_abs_scaler_fit_spmd_gold(dataframe, queue, dtype):
    from sklearnex.preview.preprocessing import MaxAbsScaler
    from sklearnex.spmd.preprocessing import MaxAbsScaler as MaxAbsScaler_SPMD

    data = np.array(
        [
            [-10.0, 0.0, 3.0],
            [2.0, -1.0, 2.0],
            [5.0, 2.0, -4.0],
            [1.0, 3.0, 8.0],
            [8.0, -4.0, 1.0],
            [-1.0, 5.0, 2.0],
            [-5.0, -6.0, 64.0],
            [2.0, 1.0, -128.0],
        ],
        dtype=dtype,
    )
    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # ensure results of batch algo match spmd
    scaler_spmd = MaxAbsScaler_SPMD().fit(local_dpt_data)
    scaler = MaxAbsScaler().fit(dpt_data)

    assert_allclose(scaler_spmd.scale_, scaler.scale_)
    assert_allclose(scaler_spmd.max_abs_, scaler.max_abs_)
    assert scaler_spmd.n_samples_seen_ == scaler.n_samples_seen_


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.mpi
def test_max_abs_scaler_partial_fit_spmd_gold(dataframe, queue, num_blocks, dtype):
    from sklearnex.preview.preprocessing import MaxAbsScaler
    from sklearnex.spmd.preprocessing import MaxAbsScaler as MaxAbsScaler_SPMD

    data = np.array(
        [
            [-1.0, 3.0, 0.0],
            [0.5, 1.0, -2.0],
            [4.0, 2.0, 4.0],
            [-3.0, -3.0, 8.0],
            [5.0, 4.0, -16.0],
            [2.0, -5.0, 32.0],
            [1.0, -6.0, -64.0],
            [-7.0, 8.0, 128.0],
        ],
        dtype=dtype,
    )
    dpt_data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)
    local_data = _get_local_tensor(data)
    split_local_data = np.array_split(local_data, num_blocks)

    scaler_spmd = MaxAbsScaler_SPMD()
    scaler = MaxAbsScaler()

    for i in range(num_blocks):
        local_dpt_data = _convert_to_dataframe(
            split_local_data[i], sycl_queue=queue, target_df=dataframe
        )
        scaler_spmd.partial_fit(local_dpt_data)

    scaler.fit(dpt_data)

    assert_allclose(scaler_spmd.scale_, scaler.scale_)
    assert_allclose(scaler_spmd.max_abs_, scaler.max_abs_)
    assert scaler_spmd.n_samples_seen_ == scaler.n_samples_seen_
