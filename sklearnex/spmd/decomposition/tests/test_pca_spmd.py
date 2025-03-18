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
@pytest.mark.mpi
def test_pca_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.decomposition import PCA as PCA_Batch
    from sklearnex.spmd.decomposition import PCA as PCA_SPMD

    # Create gold data and convert to dataframe
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
        ]
    )

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # Ensure results of batch algo match spmd
    spmd_result = PCA_SPMD(n_components=2).fit(local_dpt_data)
    batch_result = PCA_Batch(n_components=2).fit(data)

    assert_allclose(spmd_result.mean_, batch_result.mean_)
    assert_allclose(spmd_result.components_, batch_result.components_)
    assert_allclose(spmd_result.singular_values_, batch_result.singular_values_)
    assert_allclose(
        spmd_result.noise_variance_,
        batch_result.noise_variance_,
        atol=1e-7,
    )
    assert_allclose(
        spmd_result.explained_variance_ratio_, batch_result.explained_variance_ratio_
    )


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("n_components", [0.5, 3, "mle", None])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_pca_spmd_synthetic(
    n_samples, n_features, n_components, whiten, dataframe, queue, dtype, use_raw_input
):
    # TODO: Resolve issues with batch fallback and lack of support for n_rows_rank < n_cols
    if n_components == "mle" or n_components == 3:
        pytest.skip("Avoid error in case of batch fallback to sklearn")
    if n_samples <= n_features:
        pytest.skip("Avoid n_samples < n_features error from spmd data split")

    # Import spmd and batch algo
    from sklearnex.decomposition import PCA as PCA_Batch
    from sklearnex.spmd.decomposition import PCA as PCA_SPMD

    # Generate data and convert to dataframe
    data = _generate_statistic_data(n_samples, n_features, dtype=dtype)

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # Ensure results of batch algo match spmd
    with config_context(use_raw_input=use_raw_input):
        spmd_result = PCA_SPMD(n_components=n_components, whiten=whiten).fit(
            local_dpt_data
        )
    batch_result = PCA_Batch(n_components=n_components, whiten=whiten).fit(data)

    tol = 1e-3 if dtype == np.float32 else 1e-7
    assert_allclose(spmd_result.mean_, batch_result.mean_, atol=tol)
    assert_allclose(spmd_result.components_, batch_result.components_, atol=tol, rtol=tol)
    assert_allclose(spmd_result.singular_values_, batch_result.singular_values_, atol=tol)
    assert_allclose(spmd_result.noise_variance_, batch_result.noise_variance_, atol=tol)
    assert_allclose(
        spmd_result.explained_variance_ratio_,
        batch_result.explained_variance_ratio_,
        atol=tol,
    )
