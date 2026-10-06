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
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse as sp

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

_csr_array = sp.csr_array if hasattr(sp, "csr_array") else sp.csr_matrix

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
    from sklearnex.cluster import HDBSCAN

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
    from sklearnex.cluster import HDBSCAN

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
    from sklearnex.cluster import HDBSCAN

    X = _grouped_data()
    hdbscan = HDBSCAN(
        min_cluster_size=_MIN_CLUSTER_SIZE,
        cluster_selection_method=cluster_selection_method,
        store_centers=store_centers,
    ).fit(X)
    assert hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)

    if store_centers in ("centroid", "both"):
        # weighted by the membership strengths, as scikit-learn weights them
        probabilities = _as_numpy(hdbscan.probabilities_)
        expected = np.stack(
            [
                np.average(group, weights=weights, axis=0)
                for group, weights in zip(_groups(X), _groups(probabilities))
            ]
        )
        assert_allclose(_in_group_order(hdbscan.centroids_), expected, atol=1e-5)
    else:
        assert not hasattr(hdbscan, "centroids_")

    if store_centers in ("medoid", "both"):
        # a medoid is one of the samples of the cluster it represents
        for medoid, group in zip(_in_group_order(hdbscan.medoids_), _groups(X)):
            assert np.isclose(group, medoid).all(axis=1).any()
    else:
        assert not hasattr(hdbscan, "medoids_")


def test_hdbscan_centers_all_noise():
    """No cluster means no center, reported as scikit-learn reports it."""
    from sklearnex.cluster import HDBSCAN

    X = _grouped_data()
    # a cluster has to hold every sample to be kept, which none of the groups does
    hdbscan = HDBSCAN(min_cluster_size=len(X), store_centers="both").fit(X)
    assert hasattr(hdbscan, "_onedal_estimator")
    assert (_as_numpy(hdbscan.labels_) == -1).all()

    # oneDAL reports no centers at all here, while scikit-learn returns them empty
    for centers in (hdbscan.centroids_, hdbscan.medoids_):
        assert _as_numpy(centers).shape == (0, X.shape[1])
        # not a view on 'X', which would keep the whole input alive
        assert centers.base is None


def test_hdbscan_probabilities():
    """'probabilities_' must be a membership strength in the assigned cluster."""
    from sklearnex.cluster import HDBSCAN

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(_grouped_data())
    assert hasattr(hdbscan, "_onedal_estimator")

    labels = _as_numpy(hdbscan.labels_)
    probabilities = _as_numpy(hdbscan.probabilities_)
    assert probabilities.shape == labels.shape
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    # a sample belongs to its cluster to some degree, noise to none at all
    assert np.all(probabilities[labels == -1] == 0)
    assert np.all(probabilities[labels != -1] > 0)

    # the strengths are relative to the most persistent member of the cluster,
    # which therefore reaches 1
    for label in np.unique(labels[labels != -1]):
        assert_allclose(probabilities[labels == label].max(), 1.0)


def test_hdbscan_single_linkage_tree_dtype():
    """The record layout must stay the one scikit-learn's own routines are typed on."""
    from sklearn.cluster._hdbscan._tree import HIERARCHY_dtype

    from sklearnex.cluster.hdbscan import _SINGLE_LINKAGE_TREE_DTYPE

    assert _SINGLE_LINKAGE_TREE_DTYPE == HIERARCHY_dtype


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_single_linkage_tree(dataframe, queue, dtype):
    """'_single_linkage_tree_' must be a dendrogram of the data."""
    from sklearnex.cluster import HDBSCAN
    from sklearnex.cluster.hdbscan import _SINGLE_LINKAGE_TREE_DTYPE

    X = _grouped_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    n_samples = len(X)
    tree = hdbscan._single_linkage_tree_
    assert tree.dtype == _SINGLE_LINKAGE_TREE_DTYPE
    assert tree.shape == (n_samples - 1,)

    # one merge per row, in ascending distance order, each of the two merged nodes
    # either a sample or a node some earlier row created
    assert np.all(np.diff(tree["value"]) >= 0)
    for row, (left, right) in enumerate(zip(tree["left_node"], tree["right_node"])):
        assert 0 <= left < n_samples + row
        assert 0 <= right < n_samples + row
    assert tree["cluster_size"][-1] == n_samples


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_hdbscan_dbscan_clustering(dataframe, queue):
    """scikit-learn's 'dbscan_clustering' must work off oneDAL's dendrogram.

    This is the point of exporting it: the DBSCAN clustering at a given epsilon
    comes out of the hierarchy that was already built, without fitting again.
    """
    from sklearnex.cluster import HDBSCAN

    X = _grouped_data()
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    # the groups are tighter than one unit across and tens of units apart, so a cut
    # anywhere in between has to recover exactly them, and a cut above everything
    # has to put all of the samples together
    assert_groups_found(hdbscan.dbscan_clustering(cut_distance=5.0))

    joined = hdbscan.dbscan_clustering(cut_distance=1000.0)
    assert len(np.unique(joined)) == 1
    assert np.all(joined != -1)

    # nothing merges below the within-group spread, so every sample is on its own
    # and no component reaches 'min_cluster_size'
    assert np.all(hdbscan.dbscan_clustering(cut_distance=1e-6) == -1)


@pytest.mark.allow_sklearn_fallback
def test_hdbscan_sparse_falls_back():
    """Sparse data is clustered by scikit-learn, which supports it."""
    from sklearnex.cluster import HDBSCAN

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(_csr_array(_grouped_data()))
    assert not hasattr(hdbscan, "_onedal_estimator")
    assert_groups_found(hdbscan.labels_)


def _fit_both(X, X_sklearn=None, **params):
    """Fit scikit-learn's estimator and the patched one, each on its own copy.

    Returns
    -------
    expected, result : HDBSCAN or Exception
        The fitted estimators, or what each of them raised.
    """
    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN

    from sklearnex.cluster import HDBSCAN

    params.setdefault("copy", False)
    outcomes = []
    for estimator, data in (
        (_sklearn_HDBSCAN, X if X_sklearn is None else X_sklearn),
        (HDBSCAN, X),
    ):
        try:
            outcomes.append(estimator(**params).fit(data))
        except Exception as error:
            outcomes.append(error)
    return outcomes


def _assert_same_error(expected, result):
    assert isinstance(expected, Exception), "scikit-learn did not raise"
    assert type(result) is type(expected)
    assert str(result) == str(expected)


def _assert_same_partition(expected, result):
    """The same clustering, with the same special labels, in any numbering."""
    expected, result = _as_numpy(expected), _as_numpy(result)
    assert expected.shape == result.shape
    for special in (-1, -2, -3):
        assert_array_equal(expected == special, result == special)
    pairs = set(zip(expected.tolist(), result.tolist()))
    assert len(pairs) == len(set(expected.tolist())) == len(set(result.tolist()))


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize(
    "metric,metric_params",
    [
        ("euclidean", None),
        ("euclidean", {}),
        ("l2", None),
        ("manhattan", None),
        ("l1", None),
        ("cityblock", None),
        ("chebyshev", None),
        ("infinity", None),
        ("p", None),
        ("p", {"p": 3}),
        ("minkowski", None),
        ("minkowski", {"p": 1}),
        ("minkowski", {"p": 2.0}),
        ("minkowski", {"p": np.inf}),
        ("minkowski", {"p": 1.5}),
        ("minkowski", {"p": 0.5}),
        ("minkowski", {"p": 2, "w": np.ones(2)}),
        ("euclidean", {"p": 2}),
        ("cosine", None),
        ("seuclidean", None),
    ],
)
def test_hdbscan_metric_params_match_sklearn(metric, metric_params, algorithm):
    """Every metric scikit-learn computes identically is offloaded, the rest is not."""
    from sklearnex.cluster.hdbscan import _onedal_metric

    expected, result = _fit_both(
        _grouped_data(),
        min_cluster_size=_MIN_CLUSTER_SIZE,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
    )
    offloaded = _onedal_metric(metric, metric_params, algorithm) is not None
    if isinstance(expected, Exception):
        assert not offloaded
        _assert_same_error(expected, result)
        return
    assert hasattr(result, "_onedal_estimator") == offloaded
    _assert_same_partition(expected.labels_, result.labels_)


def _non_finite_data(dtype=np.float64):
    X = _grouped_data(dtype)
    X[3, 0] = np.nan
    X[17, 1] = np.inf
    X[20, 0] = -np.inf
    X[33] = [np.nan, np.inf]
    # infinities of both signs sum to NaN, which scikit-learn takes as missing
    X[40] = [np.inf, -np.inf]
    return X


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_non_finite_matches_sklearn(dataframe, queue, algorithm, dtype):
    """The samples with NaN or inf are labelled as scikit-learn labels them."""
    X = _non_finite_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    expected, result = _fit_both(
        X_df, X, min_cluster_size=_MIN_CLUSTER_SIZE, algorithm=algorithm
    )
    assert hasattr(result, "_onedal_estimator")

    labels = _as_numpy(result.labels_)
    assert labels.dtype == expected.labels_.dtype
    assert_array_equal(labels[[17, 20]], -2)
    assert_array_equal(labels[[3, 33, 40]], -3)
    _assert_same_partition(expected.labels_, labels)

    probabilities = _as_numpy(result.probabilities_)
    assert probabilities.dtype == expected.probabilities_.dtype
    assert_array_equal(np.isnan(probabilities), np.isnan(expected.probabilities_))
    # oneDAL computes in the dtype of the data, scikit-learn always in float64,
    # and the strengths are ratios of the small distances within the groups
    assert_allclose(
        probabilities,
        expected.probabilities_,
        atol=1e-5 if dtype == np.float64 else 0.1,
    )

    tree = result._single_linkage_tree_
    assert tree.dtype == expected._single_linkage_tree_.dtype
    assert tree.shape == expected._single_linkage_tree_.shape
    # the outliers are merged last, at an infinite distance, in scikit-learn's order
    n_outliers = 5
    assert_array_equal(tree[-n_outliers:], expected._single_linkage_tree_[-n_outliers:])
    assert np.all(np.isfinite(tree["value"][:-n_outliers]))

    for cut_distance in (5.0, 1000.0):
        _assert_same_partition(
            expected.dbscan_clustering(cut_distance),
            result.dbscan_clustering(cut_distance),
        )


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize(
    "X,params",
    [
        # scikit-learn fails to compute the centers of data with outliers
        (_non_finite_data(), {"store_centers": "both"}),
        # too few finite samples left for 'min_samples'
        (np.where(np.arange(45)[:, None] < 3, 1.0, np.nan) * np.ones((45, 2)), {}),
        (
            np.where(np.arange(45)[:, None] < 1, 1.0, np.inf) * np.ones((45, 2)),
            {"min_samples": 1},
        ),
        (_grouped_data()[:1], {}),
        (_grouped_data()[:1], {"min_samples": 1}),
        (_grouped_data()[:4], {}),
        (_grouped_data()[:0], {}),
        (_grouped_data()[:, :0], {}),
        (_grouped_data()[:, 0], {}),
    ],
)
def test_hdbscan_errors_match_sklearn(X, params):
    """Whatever scikit-learn rejects falls back, to raise scikit-learn's error."""
    expected, result = _fit_both(X, **params)
    _assert_same_error(expected, result)
    assert not hasattr(result, "_onedal_estimator")


@pytest.mark.parametrize(
    "params",
    [
        {"leaf_size": 0},
        {"leaf_size": 2.5},
        {"min_cluster_size": 1},
        {"min_samples": 0},
        {"alpha": 0.0},
        {"max_cluster_size": 0},
        {"cluster_selection_epsilon": -1.0},
        {"cluster_selection_method": "bad"},
        {"algorithm": "kdtree"},
        {"store_centers": "bad"},
        {"copy": "bad"},
        {"n_jobs": "bad"},
        {"metric": "bad"},
        {"metric_params": "bad"},
        {"metric": "precomputed", "store_centers": "both"},
    ],
)
@pytest.mark.allow_sklearn_fallback
def test_hdbscan_invalid_parameters_match_sklearn(params):
    expected, result = _fit_both(_grouped_data(), **params)
    _assert_same_error(expected, result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_output_dtypes_match_sklearn(dtype):
    """scikit-learn computes in float64 whatever the input, and so types its output."""
    X = _grouped_data(dtype)
    expected, result = _fit_both(
        X, min_cluster_size=_MIN_CLUSTER_SIZE, store_centers="both"
    )
    assert hasattr(result, "_onedal_estimator")
    for attribute in (
        "labels_",
        "probabilities_",
        "centroids_",
        "medoids_",
        "_single_linkage_tree_",
    ):
        assert getattr(result, attribute).dtype == getattr(expected, attribute).dtype
    assert result._min_samples == expected._min_samples
    assert result._metric_params == expected._metric_params
