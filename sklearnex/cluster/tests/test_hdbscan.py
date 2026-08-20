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
from scipy.sparse import csr_matrix
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


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_sklearnex_import_hdbscan(dataframe, queue, dtype, metric):
    """oneDAL clustering must agree with the one from scikit-learn."""
    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric).fit(X_df)
    assert "sklearnex" in hdbscan.__module__
    assert hasattr(hdbscan, "_onedal_estimator")

    expected = _sklearn_HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric)
    ari = adjusted_rand_score(expected.fit(X).labels_, _as_numpy(hdbscan.labels_))
    assert ari > 0.9, f"clustering differs from scikit-learn's: ARI={ari}"


@pytest.mark.parametrize(
    "params",
    [
        {"metric": "precomputed"},
        {"metric": "cosine", "algorithm": "kd_tree"},
        {"cluster_selection_method": "leaf_oldest"},
        {"min_cluster_size": 200},
    ],
)
def test_hdbscan_unsupported_params(params):
    """Parameters outside of oneDAL's support must fall back to scikit-learn."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    assert not HDBSCAN(**params)._onedal_supported("fit", X).get_status()


@pytest.mark.parametrize(
    "params",
    [
        {"cluster_selection_method": "leaf"},
        {"cluster_selection_epsilon": 0.5},
        {"max_cluster_size": 50},
        {"allow_single_cluster": True},
        {"store_centers": "both"},
        {"metric": "minkowski", "metric_params": {"p": 3}},
        {"algorithm": "ball_tree"},
    ],
)
def test_hdbscan_supported_params(params):
    """Parameters oneDAL supports must not trigger a fallback."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
    assert HDBSCAN(min_cluster_size=10, **params)._onedal_supported("fit", X).get_status()


@pytest.mark.parametrize("unsupported", ["sparse", "non-finite"])
def test_hdbscan_fallback_data(unsupported):
    """Data unsupported by oneDAL is clustered by scikit-learn instead."""
    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    if unsupported == "sparse":
        X = csr_matrix(X)
    else:
        X[0, 0] = np.nan

    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)
    assert not hasattr(hdbscan, "_onedal_estimator")

    expected = _sklearn_HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)
    np.testing.assert_array_equal(expected.labels_, hdbscan.labels_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("store_centers", [None, "centroid", "medoid", "both"])
def test_hdbscan_store_centers(dataframe, queue, store_centers):
    """Centers are exposed with sklearn's shapes only when requested."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(
        min_cluster_size=15, min_samples=5, store_centers=store_centers
    ).fit(X)

    labels = _as_numpy(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for attr, requested in [
        ("centroids_", store_centers in ("centroid", "both")),
        ("medoids_", store_centers in ("medoid", "both")),
    ]:
        assert hasattr(hdbscan, attr) == requested, f"unexpected state of {attr}"
        if requested:
            assert _as_numpy(getattr(hdbscan, attr)).shape == (n_clusters, 2)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_probabilities(dataframe, queue):
    """``probabilities_`` is not computed by oneDAL, but keeps namespace and device."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5).fit(X)

    probabilities = hdbscan.probabilities_
    assert probabilities.shape == hdbscan.labels_.shape
    assert np.all(_as_numpy(probabilities) == 0)
    if dataframe not in ("numpy", "pandas"):
        assert type(probabilities) is type(hdbscan.labels_)
        if hasattr(hdbscan.labels_, "device"):
            assert probabilities.device == hdbscan.labels_.device
