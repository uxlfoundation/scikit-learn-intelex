# ==============================================================================
# Copyright contributors to the oneDAL project
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

"""Tests for the onedal.cluster.HDBSCAN low-level wrapper."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from onedal.cluster import HDBSCAN
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
def test_onedal_hdbscan_basic(dataframe, queue):
    """Basic HDBSCAN fit and label output."""
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    labels = np.asarray(h.labels_)
    assert labels.shape == (200,)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
def test_onedal_hdbscan_correctness(dataframe, queue):
    """HDBSCAN should achieve high ARI on well-separated blobs."""
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    ari = adjusted_rand_score(y_true, h.labels_)
    assert ari > 0.9, f"ARI too low: {ari}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_onedal_hdbscan_metrics(dataframe, queue, metric):
    """HDBSCAN with different distance metrics."""
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric)
    h.fit(X, queue=queue)

    labels = np.asarray(h.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters with {metric}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_onedal_hdbscan_dtypes(dataframe, queue, dtype):
    """HDBSCAN with float32 and float64."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = X.astype(dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    labels = np.asarray(h.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters with {dtype}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
def test_onedal_hdbscan_algorithm_auto(dataframe, queue):
    """Algorithm='auto' should pick kd_tree for euclidean."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, algorithm="auto")
    h.fit(X, queue=queue)

    labels = np.asarray(h.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
def test_onedal_hdbscan_min_samples_none(dataframe, queue):
    """When min_samples=None, uses min_cluster_size as default."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h1 = HDBSCAN(min_cluster_size=15, min_samples=None)
    h1.fit(X, queue=queue)

    h2 = HDBSCAN(min_cluster_size=15, min_samples=15)
    h2.fit(X, queue=queue)

    ari = adjusted_rand_score(h1.labels_, h2.labels_)
    assert ari == 1.0, f"min_samples=None should match min_samples=mcs: ARI={ari}"
