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

from onedal.tests.utils._spmd_support import (
    assert_unordered_allclose,
    assert_kmeans_labels_allclose,
    generate_clustering_data,
    get_local_tensor,
    mpi_libs_and_gpu_available,
    spmd_assert_allclose,
)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_kmeans_spmd_manual():
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
        ]
    )
    X_test = np.array([[0, 0], [12, 3], [2, 2], [7, 8]])

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure labels from fit of batch algo matches spmd
    spmd_model = KMeans_SPMD(n_clusters=2, random_state=0).fit(local_dpt_X_train)
    batch_model = KMeans_Batch(n_clusters=2, random_state=0).fit(X_train)

    assert_unordered_allclose(spmd_model.cluster_centers_, batch_model.cluster_centers_)
    assert_kmeans_labels_allclose(
        spmd_model.labels_,
        batch_model.labels_,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )
    assert_allclose(spmd_model.n_iter_, batch_model.n_iter_, atol=1)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_kmeans_labels_allclose(
        spmd_result,
        batch_result,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 10000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_clusters", [2, 5, 15])
@pytest.mark.mpi
def test_kmeans_spmd_synthetic(n_samples, n_features, n_clusters):
    # Import spmd and batch algo
    from sklearnex.cluster import KMeans as KMeans_Batch
    from sklearnex.spmd.cluster import KMeans as KMeans_SPMD

    # TODO: investigate issues when centers != n_clusters (spmd and batch results don't match for all values of K)
    X_train, X_test, _, _ = generate_clustering_data(
        n_samples, n_features, centers=n_clusters
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # kmeans init
    spmd_model_init = KMeans_SPMD(n_clusters=n_clusters, max_iter=1, random_state=0).fit(
        local_dpt_X_train
    )
    batch_model_init = KMeans_Batch(
        n_clusters=n_clusters, max_iter=1, random_state=0
    ).fit(X_train)
    # TODO: centers do not match up after init
    # assert_unordered_allclose(spmd_model_init.cluster_centers_, batch_model_init.cluster_centers_)

    # ensure labels from fit of batch algo matches spmd, using same init
    spmd_model = KMeans_SPMD(
        n_clusters=n_clusters, init=spmd_model_init.cluster_centers_, random_state=0
    ).fit(local_dpt_X_train)
    batch_model = KMeans_Batch(
        n_clusters=n_clusters, init=spmd_model_init.cluster_centers_, random_state=0
    ).fit(X_train)

    assert_unordered_allclose(spmd_model.cluster_centers_, batch_model.cluster_centers_)
    assert_kmeans_labels_allclose(
        spmd_model.labels_,
        batch_model.labels_,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )
    # TODO: why are iters generally off by 1
    assert_allclose(spmd_model.n_iter_, batch_model.n_iter_, atol=1)

    # ensure predictions of batch algo match spmd
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    assert_kmeans_labels_allclose(
        spmd_result,
        batch_result,
        spmd_model.cluster_centers_,
        batch_model.cluster_centers_,
    )
