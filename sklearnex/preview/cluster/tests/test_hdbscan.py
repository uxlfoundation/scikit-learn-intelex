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
from numpy.testing import assert_allclose

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

# Three tight groups of samples, far apart from each other and lying in clearly
# different directions as seen from the origin, so that every supported metric,
# the cosine distance included, has to recover exactly these groups. The data is
# built here rather than compared against another HDBSCAN implementation: the
# conformance with scikit-learn's own results is covered by running its test
# suite against the patched estimator.
_GROUP_SIZE = 15
_CENTERS = np.array([[20.0, 1.0], [1.0, 20.0], [-20.0, -20.0]])
_MIN_CLUSTER_SIZE = 5


def _grouped_data(dtype=np.float64):
    generator = np.random.default_rng(42)
    X = np.concatenate(
        [
            center + generator.normal(scale=0.05, size=(_GROUP_SIZE, len(center)))
            for center in _CENTERS
        ]
    )
    return X.astype(dtype)


def _groups(X):
    """The samples of every group, in the order the data was built in."""
    return [X[i * _GROUP_SIZE : (i + 1) * _GROUP_SIZE] for i in range(len(_CENTERS))]


def assert_groups_found(labels):
    """Check that a clustering is exactly the grouping the data was built from.

    The clusters are the same as scikit-learn's, but the two implementations do
    not necessarily number them in the same way, so a clustering is compared
    through the grouping of the samples that it induces rather than by label.
    """
    labels = _as_numpy(labels)
    found = [set(group.tolist()) for group in _groups(labels)]
    assert all(len(group) == 1 for group in found), f"groups not recovered: {labels}"
    # every group is a cluster of its own, and nothing was labelled as noise
    assert len(set().union(*found)) == len(_CENTERS)
    assert -1 not in set().union(*found)


def _in_group_order(centers):
    """Centers in the order of the groups they belong to.

    The centers follow the numbering of the clusters, which is arbitrary.
    """
    centers = _as_numpy(centers)
    assert centers.shape == _CENTERS.shape
    # the groups are far apart, so the closest expected center is unambiguous
    order = np.argmin(
        np.linalg.norm(centers[:, None, :] - _CENTERS[None, :, :], axis=2), axis=1
    )
    assert sorted(order.tolist()) == list(range(len(_CENTERS)))
    return centers[np.argsort(order)]


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_import_hdbscan(dataframe, queue, dtype):
    """oneDAL must find the groups the data was built from."""
    from sklearnex.preview.cluster import HDBSCAN

    X = _grouped_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(X_df)
    assert "sklearnex" in hdbscan.__module__
    assert hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)


# scikit-learn's own tests only cover the default 'algorithm', while every
# combination below maps onto a different oneDAL method
@pytest.mark.parametrize(
    "metric,algorithm,metric_params",
    [
        ("euclidean", "auto", None),
        ("euclidean", "brute", None),
        ("manhattan", "kd_tree", None),
        ("chebyshev", "ball_tree", None),
        ("minkowski", "kd_tree", {"p": 3}),
        ("cosine", "brute", None),
    ],
)
def test_hdbscan_metrics(metric, algorithm, metric_params):
    """Every metric and algorithm offloaded to oneDAL must find the groups."""
    from sklearnex.preview.cluster import HDBSCAN

    hdbscan = HDBSCAN(
        min_cluster_size=_MIN_CLUSTER_SIZE,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
    ).fit(_grouped_data())
    assert hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)


@pytest.mark.parametrize("cluster_selection_method", ["eom", "leaf"])
@pytest.mark.parametrize("store_centers", ["centroid", "medoid", "both"])
def test_hdbscan_centers(cluster_selection_method, store_centers):
    """oneDAL computes the centers only when it is asked to."""
    from sklearnex.preview.cluster import HDBSCAN

    X = _grouped_data()
    hdbscan = HDBSCAN(
        min_cluster_size=_MIN_CLUSTER_SIZE,
        cluster_selection_method=cluster_selection_method,
        store_centers=store_centers,
    ).fit(X)
    assert hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)

    if store_centers in ("centroid", "both"):
        expected = np.stack([group.mean(axis=0) for group in _groups(X)])
        assert_allclose(_in_group_order(hdbscan.centroids_), expected, atol=1e-5)
    else:
        assert not hasattr(hdbscan, "centroids_")

    if store_centers in ("medoid", "both"):
        # a medoid is one of the samples of the cluster it represents
        for medoid, group in zip(_in_group_order(hdbscan.medoids_), _groups(X)):
            assert np.isclose(group, medoid).all(axis=1).any()
    else:
        assert not hasattr(hdbscan, "medoids_")


def test_hdbscan_probabilities():
    """'probabilities_' must stay a probability."""
    from sklearnex.preview.cluster import HDBSCAN

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(_grouped_data())
    assert hasattr(hdbscan, "_onedal_estimator")

    probabilities = _as_numpy(hdbscan.probabilities_)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    # oneDAL does not return the lambda values that the membership strengths are
    # derived from, so a sample either belongs to its cluster or is noise
    assert_allclose(probabilities, _as_numpy(hdbscan.labels_) != -1)


def test_hdbscan_sparse_falls_back():
    """Sparse data is clustered by scikit-learn, which supports it."""
    from scipy.sparse import csr_matrix

    from sklearnex.preview.cluster import HDBSCAN

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(csr_matrix(_grouped_data()))
    assert not hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)
