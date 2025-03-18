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

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils.spmd import (
    _generate_clustering_data,
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
def test_dbscan_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.cluster import DBSCAN as DBSCAN_Batch
    from sklearnex.spmd.cluster import DBSCAN as DBSCAN_SPMD

    data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # Ensure labels from fit of batch algo matches spmd
    spmd_model = DBSCAN_SPMD(eps=3, min_samples=2).fit(local_dpt_data)
    batch_model = DBSCAN_Batch(eps=3, min_samples=2).fit(data)

    _spmd_assert_allclose(spmd_model.labels_, batch_model.labels_)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features_and_eps", [(5, 3), (5, 10), (25, 10)])
@pytest.mark.parametrize("centers", [10, None])
@pytest.mark.parametrize("min_samples", [2, 5, 15])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_dbscan_spmd_synthetic(
    n_samples,
    n_features_and_eps,
    centers,
    min_samples,
    dataframe,
    queue,
    dtype,
    use_raw_input,
):
    n_features, eps = n_features_and_eps
    # Import spmd and batch algo
    from sklearnex.cluster import DBSCAN as DBSCAN_Batch
    from sklearnex.spmd.cluster import DBSCAN as DBSCAN_SPMD

    data, _ = _generate_clustering_data(
        n_samples, n_features, centers=centers, dtype=dtype
    )

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # Ensure labels from fit of batch algo matches spmd
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_model = DBSCAN_SPMD(eps=eps, min_samples=min_samples).fit(local_dpt_data)
    batch_model = DBSCAN_Batch(eps=eps, min_samples=min_samples).fit(data)

    _spmd_assert_allclose(spmd_model.labels_, batch_model.labels_)

    # Ensure meaningful test setup
    if np.all(batch_model.labels_ == -1):
        raise ValueError("No labels given - try raising epsilon")
