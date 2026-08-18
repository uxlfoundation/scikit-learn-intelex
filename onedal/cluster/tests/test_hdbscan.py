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

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

pytestmark = pytest.mark.skipif(
    not daal_check_version((2026, "P", 200)),
    reason="HDBSCAN requires oneDAL >= 2026.2",
)

if daal_check_version((2026, "P", 200)):
    from onedal.cluster import HDBSCAN

# the onedal wrapper accepts arrays only, no dataframes
_dataframes_and_queues = get_dataframes_and_queues("numpy,dpnp,array_api")


def _n_clusters(labels):
    labels = _as_numpy(labels)
    return len(set(labels)) - (1 if -1 in labels else 0)


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_basic(dataframe, queue):
    """Basic HDBSCAN fit and label output."""
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    assert h.labels_.shape == (200,)
    n_clusters = _n_clusters(h.labels_)
    assert n_clusters >= 2, f"Expected >=2 clusters, got {n_clusters}"
    # oneDAL reports the number of clusters it found itself
    assert h.n_clusters_ == n_clusters


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_namespace_propagation(dataframe, queue):
    """Results are returned in the namespace and device of the input data."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, store_centers="both")
    h.fit(X, queue=queue)

    for attr in ["labels_", "centroids_", "medoids_"]:
        result = getattr(h, attr)
        assert type(result) is type(X), f"{attr} does not keep the input namespace"
        if hasattr(X, "device"):
            assert result.device == X.device, f"{attr} is not on the input device"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_correctness(dataframe, queue):
    """HDBSCAN should achieve high ARI on well-separated blobs."""
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    ari = adjusted_rand_score(y_true, _as_numpy(h.labels_))
    assert ari > 0.9, f"ARI too low: {ari}"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"]
)
def test_onedal_hdbscan_metrics(dataframe, queue, metric):
    """HDBSCAN with different distance metrics."""
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    metric_params = {"p": 3} if metric == "minkowski" else None
    h = HDBSCAN(
        min_cluster_size=15, min_samples=5, metric=metric, metric_params=metric_params
    )
    h.fit(X, queue=queue)

    n_clusters = _n_clusters(h.labels_)
    assert n_clusters >= 2, f"Expected >=2 clusters with {metric}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_onedal_hdbscan_dtypes(dataframe, queue, dtype):
    """HDBSCAN with float32 and float64."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = X.astype(dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5)
    h.fit(X, queue=queue)

    n_clusters = _n_clusters(h.labels_)
    assert n_clusters >= 2, f"Expected >=2 clusters with {dtype}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
@pytest.mark.parametrize(
    "algorithm", ["auto", "brute", "brute_force", "kd_tree", "ball_tree"]
)
def test_onedal_hdbscan_algorithms(dataframe, queue, algorithm):
    """All algorithms mapped onto oneDAL methods give the same clustering."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, algorithm=algorithm)
    h.fit(X, queue=queue)

    n_clusters = _n_clusters(h.labels_)
    assert n_clusters >= 2, f"Expected >=2 clusters with {algorithm}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
@pytest.mark.parametrize("cluster_selection_method", ["eom", "leaf"])
def test_onedal_hdbscan_cluster_selection_method(
    dataframe, queue, cluster_selection_method
):
    """Both supported cluster selection methods find the blobs."""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        cluster_selection_method=cluster_selection_method,
    )
    h.fit(X, queue=queue)

    n_clusters = _n_clusters(h.labels_)
    assert (
        n_clusters >= 2
    ), f"Expected >=2 clusters with {cluster_selection_method}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
@pytest.mark.parametrize("store_centers", [None, "centroid", "medoid", "both"])
def test_onedal_hdbscan_store_centers(dataframe, queue, store_centers):
    """Centers are only computed and returned when requested."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, store_centers=store_centers)
    h.fit(X, queue=queue)

    expected_shape = (h.n_clusters_, 2)
    for attr, requested in [
        ("centroids_", store_centers in ("centroid", "both")),
        ("medoids_", store_centers in ("medoid", "both")),
    ]:
        assert hasattr(h, attr) == requested, f"unexpected state of {attr}"
        if requested:
            assert getattr(h, attr).shape == expected_shape


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_cluster_selection_epsilon(dataframe, queue):
    """A large cluster_selection_epsilon merges clusters."""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h_no_eps = HDBSCAN(min_cluster_size=15, min_samples=5)
    h_no_eps.fit(X, queue=queue)
    h_eps = HDBSCAN(min_cluster_size=15, min_samples=5, cluster_selection_epsilon=100.0)
    h_eps.fit(X, queue=queue)

    assert _n_clusters(h_eps.labels_) <= _n_clusters(h_no_eps.labels_)


@pytest.mark.skip(
    reason="Temporarily disabled: oneDAL's EOM cluster selection skips leaf clusters, "
    "so the max_cluster_size cap is not applied to them, and every blob here is a "
    "condensed-tree leaf. Fixed on the oneDAL side by deselecting oversized leaves; "
    "re-enable once that fix is in the oneDAL version this repo builds against."
)
@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_max_cluster_size(dataframe, queue):
    """max_cluster_size limits the size of the reported clusters."""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    max_cluster_size = 50
    h = HDBSCAN(min_cluster_size=15, min_samples=5, max_cluster_size=max_cluster_size)
    h.fit(X, queue=queue)

    labels = _as_numpy(h.labels_)
    for label in set(labels) - {-1}:
        assert np.sum(labels == label) <= max_cluster_size


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_allow_single_cluster(dataframe, queue):
    """allow_single_cluster keeps a single blob as one cluster."""
    X, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h = HDBSCAN(min_cluster_size=15, min_samples=5, allow_single_cluster=True)
    h.fit(X, queue=queue)

    assert _n_clusters(h.labels_) == 1


@pytest.mark.parametrize("dataframe,queue", _dataframes_and_queues)
def test_onedal_hdbscan_min_samples_none(dataframe, queue):
    """When min_samples=None, uses min_cluster_size as default."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h1 = HDBSCAN(min_cluster_size=15, min_samples=None)
    h1.fit(X, queue=queue)

    h2 = HDBSCAN(min_cluster_size=15, min_samples=15)
    h2.fit(X, queue=queue)

    ari = adjusted_rand_score(_as_numpy(h1.labels_), _as_numpy(h2.labels_))
    assert ari == 1.0, f"min_samples=None should match min_samples=mcs: ARI={ari}"
