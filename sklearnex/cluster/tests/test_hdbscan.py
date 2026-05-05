# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

pytestmark = pytest.mark.skipif(
    not sklearn_check_version("1.3"),
    reason="HDBSCAN requires sklearn >= 1.3",
)


# ============================================================================
# Basic functionality tests
# ============================================================================


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_hdbscan(dataframe, queue):
    """Verify that HDBSCAN is importable from sklearnex and patching works."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
    assert "sklearnex" in hdbscan.__module__


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_three_clusters(dataframe, queue):
    """Well-separated clusters should have ARI > 0.9 against ground truth."""
    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)

    ari = adjusted_rand_score(y_true, hdbscan.labels_)
    assert ari > 0.9, f"ARI too low: {ari}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_labels_shape(dataframe, queue):
    """Verify output attributes have correct shapes and types."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10).fit(X)

    assert hasattr(hdbscan, "labels_")
    labels = np.asarray(hdbscan.labels_)
    assert labels.shape == (200,)
    assert hasattr(hdbscan, "probabilities_")
    assert hasattr(hdbscan, "n_features_in_")
    assert hdbscan.n_features_in_ == 2


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_fit_returns_self(dataframe, queue):
    """Verify fit() returns self for method chaining."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=5)
    result = hdbscan.fit(X)
    assert result is hdbscan


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_fit_predict(dataframe, queue):
    """Verify fit_predict returns the same labels as fit().labels_."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan1 = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)
    hdbscan2 = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels_fp = hdbscan2.fit_predict(X)

    ari = adjusted_rand_score(hdbscan1.labels_, labels_fp)
    assert ari > 0.99, f"fit_predict labels differ from fit labels: ARI={ari}"


# ============================================================================
# Correctness tests — comparison with sklearn
# ============================================================================


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_vs_sklearn(dataframe, queue):
    """Compare sklearnex HDBSCAN against stock sklearn for correctness."""
    from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    sklearnex_hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X_df)
    sklearn_hdbscan = sklearn_HDBSCAN(min_cluster_size=15, min_samples=5, copy=True).fit(
        X
    )

    ari = adjusted_rand_score(sklearnex_hdbscan.labels_, sklearn_hdbscan.labels_)
    assert ari > 0.9, f"ARI vs sklearn: {ari}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("n_samples,n_centers", [(200, 2), (500, 5), (1000, 3)])
def test_hdbscan_vs_sklearn_various_sizes(dataframe, queue, n_samples, n_centers):
    """Compare against sklearn across different dataset sizes and cluster counts."""
    from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(
        n_samples=n_samples, centers=n_centers, cluster_std=0.5, random_state=42
    )
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    mcs = max(15, n_samples // 50)
    sklearnex_hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples=5).fit(X_df)
    sklearn_hdbscan = sklearn_HDBSCAN(min_cluster_size=mcs, min_samples=5, copy=True).fit(
        X
    )

    ari = adjusted_rand_score(sklearnex_hdbscan.labels_, sklearn_hdbscan.labels_)
    assert ari > 0.85, f"ARI vs sklearn too low for n={n_samples}, k={n_centers}: {ari}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_vs_sklearn_high_dim(dataframe, queue):
    """Compare against sklearn on high-dimensional data."""
    from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(
        n_samples=300, n_features=10, centers=3, cluster_std=1.0, random_state=42
    )
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    sklearnex_hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X_df)
    sklearn_hdbscan = sklearn_HDBSCAN(min_cluster_size=15, min_samples=5, copy=True).fit(
        X
    )

    ari = adjusted_rand_score(sklearnex_hdbscan.labels_, sklearn_hdbscan.labels_)
    assert ari > 0.85, f"ARI vs sklearn on 10d data: {ari}"


# ============================================================================
# Distance metric tests
# ============================================================================


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "metric",
    ["euclidean", "manhattan", "chebyshev", "cosine"],
)
def test_hdbscan_metrics(dataframe, queue, metric):
    """Verify that each supported metric finds reasonable clusters."""
    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric).fit(X)

    labels = np.asarray(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters with {metric}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_minkowski(dataframe, queue):
    """Verify Minkowski metric with custom p."""
    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(
        min_cluster_size=15, min_samples=5, metric="minkowski", metric_params={"p": 3}
    ).fit(X)

    labels = np.asarray(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters with minkowski(p=3), got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_hdbscan_metric_vs_sklearn(dataframe, queue, metric):
    """Compare metric results against sklearn."""
    from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    sklearnex_h = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric).fit(X_df)
    sklearn_h = sklearn_HDBSCAN(
        min_cluster_size=15, min_samples=5, metric=metric, copy=True
    ).fit(X)

    ari = adjusted_rand_score(sklearnex_h.labels_, sklearn_h.labels_)
    assert ari > 0.85, f"ARI vs sklearn with {metric}: {ari}"


# ============================================================================
# Patching and fallback tests
# ============================================================================


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_fallback_unsupported_metric(dataframe, queue):
    """Unsupported metric should not be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, metric="precomputed")
    status = hdbscan._onedal_supported("fit", X)
    assert not status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_leaf_method(dataframe, queue):
    """cluster_selection_method='leaf' should be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, cluster_selection_method="leaf")
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_epsilon(dataframe, queue):
    """Non-zero cluster_selection_epsilon should be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.5)
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_max_cluster_size(dataframe, queue):
    """max_cluster_size should be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, max_cluster_size=50)
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_allow_single_cluster(dataframe, queue):
    """allow_single_cluster=True should be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, allow_single_cluster=True)
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_store_centers(dataframe, queue):
    """store_centers should be supported by oneDAL."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, store_centers="centroid")
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_supported_euclidean(dataframe, queue):
    """Euclidean metric with default params should be supported."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=10, metric="euclidean")
    status = hdbscan._onedal_supported("fit", X)
    assert status.get_status()


# ============================================================================
# Patching system tests
# ============================================================================


def test_patch_sklearn_hdbscan():
    """Verify that patch_sklearn enables the sklearnex HDBSCAN."""
    from sklearnex import patch_sklearn, unpatch_sklearn

    try:
        patch_sklearn("sklearn.cluster.HDBSCAN")
        from sklearn.cluster import HDBSCAN

        X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=42)
        h = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
        assert (
            "sklearnex" in h.__module__
        ), f"Expected sklearnex module, got {h.__module__}"
    finally:
        unpatch_sklearn("sklearn.cluster.HDBSCAN")


def test_patch_unpatch_hdbscan():
    """Verify that unpatch restores the original sklearn HDBSCAN."""
    from sklearnex import patch_sklearn, sklearn_is_patched, unpatch_sklearn

    try:
        patch_sklearn("sklearn.cluster.HDBSCAN")
        assert sklearn_is_patched("sklearn.cluster.HDBSCAN")

        unpatch_sklearn("sklearn.cluster.HDBSCAN")
        assert not sklearn_is_patched("sklearn.cluster.HDBSCAN")
    finally:
        unpatch_sklearn("sklearn.cluster.HDBSCAN")


# ============================================================================
# Edge case and dtype tests
# ============================================================================


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_dtypes(dataframe, queue, dtype):
    """Verify HDBSCAN works with both float32 and float64."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = X.astype(dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)

    labels = np.asarray(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2, f"Expected >=2 clusters with {dtype}, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_noise_labels(dataframe, queue):
    """Verify that noise points get label -1."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=42)
    # Add outliers far from clusters
    rng = np.random.RandomState(42)
    outliers = rng.uniform(-20, 30, size=(10, 2))
    X = np.vstack([X, outliers])

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)

    labels = np.asarray(hdbscan.labels_)
    assert -1 in labels, "Expected some noise points (label=-1)"
    n_clusters = len(set(labels)) - 1  # excluding -1
    assert n_clusters >= 2, f"Expected >=2 clusters, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_large_min_cluster_size(dataframe, queue):
    """With very large min_cluster_size, expect at most 1 real cluster."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    # min_cluster_size bigger than any cluster
    hdbscan = HDBSCAN(min_cluster_size=40).fit(X)

    labels = np.asarray(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters <= 1, f"Expected <=1 cluster with large mcs, got {n_clusters}"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_cluster_selection_epsilon(dataframe, queue):
    """Verify cluster_selection_epsilon runs through oneDAL and merges clusters."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    h_no_eps = HDBSCAN(
        min_cluster_size=15, min_samples=5, cluster_selection_epsilon=0.0
    ).fit(X)
    h_eps = HDBSCAN(
        min_cluster_size=15, min_samples=5, cluster_selection_epsilon=100.0
    ).fit(X)

    labels_no_eps = np.asarray(h_no_eps.labels_)
    labels_eps = np.asarray(h_eps.labels_)
    n_no_eps = len(set(labels_no_eps)) - (1 if -1 in labels_no_eps else 0)
    n_eps = len(set(labels_eps)) - (1 if -1 in labels_eps else 0)
    assert n_eps <= n_no_eps, "Large epsilon should merge clusters"


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_max_cluster_size(dataframe, queue):
    """Verify max_cluster_size runs through oneDAL without error."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5, max_cluster_size=50).fit(X)
    assert hasattr(hdbscan, "labels_")


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_alpha(dataframe, queue):
    """Verify alpha parameter runs through oneDAL without error."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5, alpha=1.5).fit(X)
    assert hasattr(hdbscan, "labels_")


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_leaf_size(dataframe, queue):
    """Verify leaf_size parameter runs through oneDAL without error."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5, leaf_size=20).fit(X)
    assert hasattr(hdbscan, "labels_")


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_min_samples_default(dataframe, queue):
    """When min_samples=None, it should default to min_cluster_size."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    # min_samples=None (default) should use min_cluster_size
    h1 = HDBSCAN(min_cluster_size=15, min_samples=None).fit(X)
    h2 = HDBSCAN(min_cluster_size=15, min_samples=15).fit(X)

    ari = adjusted_rand_score(h1.labels_, h2.labels_)
    assert ari == 1.0, f"min_samples=None should equal min_cluster_size: ARI={ari}"
