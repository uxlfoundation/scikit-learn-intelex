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
    _assert_kmeans_labels_allclose,
    _assert_unordered_allclose,
    _generate_clustering_data,
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
def test_kmeans_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.cluster import KMeans as KMeans_Batch
    from sklearnex.spmd.cluster import KMeans as KMeans_SPMD

    X_train = np.array(
        [
            [1, 2],
            [2, 2],
            [2, 3],
            [8, 7],
            [8, 8],
            [25, 80],
            [5, 65],
            [2, 8],
            [1, 3],
            [2, 2],
            [1, 3],
            [2, 2],
        ],
        dtype=np.float64,
    )
    X_test = np.array([[0, 0], [12, 3], [2, 2], [7, 8]], dtype=np.float64)

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Ensure labels from fit of batch algo matches spmd
    spmd_model = KMeans_SPMD(n_clusters=2, random_state=0).fit(local_dpt_X_train)
    batch_model = KMeans_Batch(n_clusters=2, random_state=0).fit(X_train)

    _assert_unordered_allclose(spmd_model.cluster_centers_, batch_model.cluster_centers_)
    _assert_kmeans_labels_allclose(
        spmd_model.labels_,
        batch_model.labels_,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )
    assert_allclose(spmd_model.n_iter_, batch_model.n_iter_, atol=1)

    # Ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _assert_kmeans_labels_allclose(
        spmd_result,
        batch_result,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_clusters", [2, 5, 15])
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,dpctl", device_filter_="gpu"),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_raw_input", [True, False])
@pytest.mark.mpi
def test_kmeans_spmd_synthetic(
    n_samples, n_features, n_clusters, dataframe, queue, dtype, use_raw_input
):
    # Import spmd and batch algo
    from sklearnex.cluster import KMeans as KMeans_Batch
    from sklearnex.spmd.cluster import KMeans as KMeans_SPMD

    # TODO: investigate issues when centers != n_clusters (spmd and batch results don't match for all values of K)
    X_train, X_test = _generate_clustering_data(
        n_samples, n_features, centers=n_clusters, dtype=dtype
    )

    local_dpt_X_train = _convert_to_dataframe(
        _get_local_tensor(X_train), sycl_queue=queue, target_df=dataframe
    )
    local_dpt_X_test = _convert_to_dataframe(
        _get_local_tensor(X_test), sycl_queue=queue, target_df=dataframe
    )

    # Validate KMeans init
    spmd_model_init = KMeans_SPMD(n_clusters=n_clusters, max_iter=1, random_state=0).fit(
        local_dpt_X_train
    )
    batch_model_init = KMeans_Batch(
        n_clusters=n_clusters, max_iter=1, random_state=0
    ).fit(X_train)
    # TODO: centers do not match up after init
    # _assert_unordered_allclose(spmd_model_init.cluster_centers_, batch_model_init.cluster_centers_)

    # Ensure labels from fit of batch algo matches spmd, using same init
    spmd_model = KMeans_SPMD(
        n_clusters=n_clusters, init=spmd_model_init.cluster_centers_, random_state=0
    )
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_model.fit(local_dpt_X_train)
    batch_model = KMeans_Batch(
        n_clusters=n_clusters, init=spmd_model_init.cluster_centers_, random_state=0
    ).fit(X_train)

    atol = 1e-5 if dtype == np.float32 else 1e-7
    _assert_unordered_allclose(
        spmd_model.cluster_centers_, batch_model.cluster_centers_, atol=atol
    )
    _assert_kmeans_labels_allclose(
        spmd_model.labels_,
        batch_model.labels_,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
        atol=atol,
    )
    # TODO: KMeans iterations are not aligned
    # assert_allclose(spmd_model.n_iter_, batch_model.n_iter_, atol=1)

    # Ensure predictions of batch algo match spmd
    # Configure raw input status for spmd estimator
    with config_context(use_raw_input=use_raw_input):
        spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    _assert_kmeans_labels_allclose(
        spmd_result,
        batch_result,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
        atol=atol,
    )
