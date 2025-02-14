# ===============================================================================
# Copyright 2021 Intel Corporation
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
    get_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils import _IS_INTEL


def generate_dense_dataset(n_samples, n_features, density, n_clusters):
    np.random.seed(2024 + n_samples + n_features + n_clusters)
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    mask = np.random.binomial(1, density, (n_samples, n_features))
    X = X * mask
    return X


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "algorithm", ["lloyd" if sklearn_check_version("1.1") else "full", "elkan"]
)
@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sklearnex_import_for_dense_data(dataframe, queue, algorithm, init):
    from sklearnex.cluster import KMeans

    X_dense = generate_dense_dataset(1000, 10, 0.5, 3)
    X_dense_df = _convert_to_dataframe(X_dense, sycl_queue=queue, target_df=dataframe)

    kmeans_dense = KMeans(
        n_clusters=3, random_state=0, algorithm=algorithm, init=init
    ).fit(X_dense_df)

    if daal_check_version((2023, "P", 200)):
        assert "sklearnex" in kmeans_dense.__module__
    else:
        assert "daal4py" in kmeans_dense.__module__


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 700)),
    reason="Sparse data requires oneDAL>=2024.7.0",
)
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize(
    "algorithm", ["lloyd" if sklearn_check_version("1.1") else "full", "elkan"]
)
@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sklearnex_import_for_sparse_data(queue, algorithm, init):
    from sklearnex.cluster import KMeans

    X_dense = generate_dense_dataset(1000, 10, 0.5, 3)
    X_sparse = csr_matrix(X_dense)

    kmeans_sparse = KMeans(
        n_clusters=3, random_state=0, algorithm=algorithm, init=init
    ).fit(X_sparse)

    assert "sklearnex" in kmeans_sparse.__module__


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "algorithm", ["lloyd" if sklearn_check_version("1.1") else "full", "elkan"]
)
def test_results_on_dense_gold_data(dataframe, queue, algorithm):
    from sklearnex.cluster import KMeans

    X_train = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    X_test = np.array([[0, 0], [12, 3]])
    X_train_df = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    X_test_df = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    kmeans = KMeans(n_clusters=2, random_state=0, algorithm=algorithm).fit(X_train_df)

    if queue and queue.sycl_device.is_gpu:
        # KMeans Init Dense GPU implementation is different from CPU
        expected_cluster_labels = np.array([0, 1], dtype=np.int32)
        expected_cluster_centers = np.array([[1.0, 2.0], [10.0, 2.0]], dtype=np.float32)
        expected_inertia = 16.0
    else:
        expected_cluster_labels = np.array([1, 0], dtype=np.int32)
        expected_cluster_centers = np.array([[10.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        expected_inertia = 16.0

    assert_allclose(expected_cluster_labels, _as_numpy(kmeans.predict(X_test_df)))
    assert_allclose(expected_cluster_centers, _as_numpy(kmeans.cluster_centers_))
    assert expected_inertia == kmeans.inertia_


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 700)),
    reason="Sparse data requires oneDAL>=2024.7.0",
)
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("init", ["k-means++", "random", "arraylike"])
@pytest.mark.parametrize(
    "algorithm", ["lloyd" if sklearn_check_version("1.1") else "full", "elkan"]
)
@pytest.mark.parametrize(
    "dims", [(1000, 10, 0.95, 3), (50000, 100, 0.75, 10), (10000, 10, 0.8, 5)]
)
def test_dense_vs_sparse(queue, init, algorithm, dims):
    from sklearnex.cluster import KMeans

    if init == "random" or (not _IS_INTEL and init == "k-means++"):
        pytest.skip(f"{init} initialization for sparse K-means is non-conformant.")

    # For higher level of sparsity (smaller density) the test may fail
    n_samples, n_features, density, n_clusters = dims
    X_dense = generate_dense_dataset(n_samples, n_features, density, n_clusters)
    X_sparse = csr_matrix(X_dense)

    if init == "arraylike":
        np.random.seed(2024 + n_samples + n_features + n_clusters)
        init = X_dense[np.random.choice(n_samples, size=n_clusters, replace=False)]

    kmeans_dense = KMeans(
        n_clusters=n_clusters, random_state=0, init=init, algorithm=algorithm
    ).fit(X_dense)
    kmeans_sparse = KMeans(
        n_clusters=n_clusters, random_state=0, init=init, algorithm=algorithm
    ).fit(X_sparse)

    assert_allclose(
        kmeans_dense.cluster_centers_,
        kmeans_sparse.cluster_centers_,
    )
