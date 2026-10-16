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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

pytestmark = pytest.mark.skipif(
    not daal_check_version((2026, "P", 300)),
    reason="HDBSCAN requires oneDAL >= 2026.3",
)

_csr_array = sp.csr_array if hasattr(sp, "csr_array") else sp.csr_matrix

# Three tight groups, far apart and in different directions from the origin, so
# that every supported metric, cosine included, has to recover exactly them.
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
    from sklearnex.preview.cluster import HDBSCAN

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
    from sklearnex.preview.cluster import HDBSCAN

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

    from sklearnex.preview.cluster.hdbscan import _SINGLE_LINKAGE_TREE_DTYPE

    assert _SINGLE_LINKAGE_TREE_DTYPE == HIERARCHY_dtype


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_single_linkage_tree(dataframe, queue, dtype):
    """'_single_linkage_tree_' must be a dendrogram of the data."""
    from sklearnex.preview.cluster import HDBSCAN
    from sklearnex.preview.cluster.hdbscan import _SINGLE_LINKAGE_TREE_DTYPE

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
    from sklearnex.preview.cluster import HDBSCAN

    X = _grouped_data()
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=_MIN_CLUSTER_SIZE).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    # the groups are tighter than one unit across and tens of units apart, so a cut
    # anywhere in between has to recover exactly them, and a cut above everything
    # has to put all of the samples together
    assert_groups_found(_as_numpy(hdbscan.dbscan_clustering(cut_distance=5.0)))

    joined = _as_numpy(hdbscan.dbscan_clustering(cut_distance=1000.0))
    assert len(np.unique(joined)) == 1
    assert np.all(joined != -1)

    # nothing merges below the within-group spread, so every sample is on its own
    # and no component reaches 'min_cluster_size'
    assert np.all(_as_numpy(hdbscan.dbscan_clustering(cut_distance=1e-6)) == -1)


@pytest.mark.allow_sklearn_fallback
def test_hdbscan_sparse_falls_back():
    """Sparse data is clustered by scikit-learn, which supports it."""
    from sklearnex.preview.cluster import HDBSCAN

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

    from sklearnex.preview.cluster import HDBSCAN

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
    from sklearnex.preview.cluster.hdbscan import _onedal_metric

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
    assert probabilities.dtype == dtype
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


def _hierarchy_data(dtype=np.float64):
    """Groups of different sizes and spreads, and scattered noise around them.

    Unlike '_grouped_data', its merge distances spread over a wide range, so that
    every parameter of the cluster selection has merges to act on.
    """
    generator = np.random.default_rng(0)
    groups = [
        ((0, 0, 0), 0.3, 60),
        ((3, 0, 1), 0.6, 60),
        ((0, 4, -1), 1.0, 60),
        ((4, 4, 4), 0.4, 40),
    ]
    X = np.concatenate(
        [generator.normal(center, scale, size=(n, 3)) for center, scale, n in groups]
        + [generator.uniform(-3, 7, size=(20, 3))]
    )
    return X.astype(dtype)


# The merge distances of two oneDAL fits computed with the same arithmetic. The
# float64 bound leaves room for the order of the reductions.
_RTOL = {np.float32: 5e-6, np.float64: 1e-10}
# oneDAL's float32 brute force expands the squared distance into norms and a dot
# product, which loses digits on the smallest distances; scikit-learn and the
# trees subtract the coordinates.
_RTOL_BRUTE = {np.float32: 2e-3, np.float64: 1e-10}


def _merge_distances(hdbscan):
    return np.sort(hdbscan._single_linkage_tree_["value"])


def _partitions(tree, cuts):
    """The connected components of a dendrogram below each of the cut distances.

    Unlike the rows of the dendrogram, which order tied merges arbitrarily, these
    are the same for any two dendrograms of the same hierarchy.
    """
    n_samples = len(tree) + 1
    members = {i: [i] for i in range(n_samples)}
    component = np.arange(n_samples)
    partitions, row = [], 0
    for cut in cuts:
        while row < len(tree) and tree["value"][row] < cut:
            node = n_samples + row
            members[node] = members.pop(tree["left_node"][row]) + members.pop(
                tree["right_node"][row]
            )
            component[members[node]] = node
            row += 1
        # numbered by first sample, so that equal partitions compare equal
        _, first, inverse = np.unique(component, return_index=True, return_inverse=True)
        partitions.append(np.argsort(np.argsort(first))[inverse])
    return partitions


def _assert_same_hierarchy(expected, result, rtol, atol=0.0):
    """Two dendrograms with the same merge distances and the same partitions.

    The partitions are compared halfway between consecutive merge distances that
    are further apart than the tolerance, so a rounding error cannot move a merge
    across a cut.
    """
    distances = np.sort(expected["value"])
    assert_allclose(np.sort(result["value"]), distances, rtol=rtol, atol=atol)
    gaps = np.flatnonzero(np.diff(distances) > 4 * (rtol * distances[1:] + atol))
    cuts = (distances[gaps] + distances[gaps + 1]) / 2
    for expected_partition, partition in zip(
        _partitions(expected, cuts), _partitions(result, cuts)
    ):
        assert_array_equal(partition, expected_partition)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("alpha", [0.5, 2.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_alpha_matches_sklearn(dataframe, queue, algorithm, alpha, dtype):
    """'alpha' must scale the distances the way scikit-learn's same algorithm does."""
    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    expected, result = _fit_both(X_df, X, algorithm=algorithm, alpha=alpha)
    unscaled = HDBSCAN(algorithm=algorithm, copy=False).fit(X_df)
    assert hasattr(result, "_onedal_estimator")

    rtol = (_RTOL_BRUTE if algorithm == "brute" else _RTOL)[dtype]
    _assert_same_hierarchy(
        expected._single_linkage_tree_, result._single_linkage_tree_, rtol
    )

    distances, unscaled_distances = _merge_distances(result), _merge_distances(unscaled)
    if algorithm == "brute":
        # scikit-learn's brute force scales the core distances too, so every merge
        assert_allclose(distances, unscaled_distances / alpha, rtol=_RTOL[dtype])
    else:
        # its trees only scale the distances between samples, not the core ones
        assert not np.allclose(distances, unscaled_distances)
        assert not np.allclose(distances, unscaled_distances / alpha)


def _tree_to_labels(hdbscan):
    """scikit-learn's cluster selection, run on the estimator's own dendrogram."""
    try:
        from sklearn.cluster._hdbscan._tree import HIERARCHY_dtype, tree_to_labels
    except ImportError:
        pytest.skip("scikit-learn moved its private 'tree_to_labels'")

    try:
        return tree_to_labels(
            hdbscan._single_linkage_tree_.astype(HIERARCHY_dtype),
            hdbscan.min_cluster_size,
            hdbscan.cluster_selection_method,
            hdbscan.allow_single_cluster,
            hdbscan.cluster_selection_epsilon,
            hdbscan.max_cluster_size,
        )
    except TypeError:
        # https://github.com/scikit-learn/scikit-learn/issues/33219
        if hdbscan.cluster_selection_epsilon == 0:
            raise
        pytest.skip("scikit-learn's 'cluster_selection_epsilon' search fails here")


def _same_partition(expected, result):
    try:
        _assert_same_partition(expected, result)
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "cluster_selection_method,params,changes",
    [
        ("eom", {"max_cluster_size": 50}, True),
        # scikit-learn applies 'max_cluster_size' to the excess of mass only
        ("leaf", {"max_cluster_size": 50}, False),
        ("eom", {"cluster_selection_epsilon": 2.0}, True),
        ("leaf", {"cluster_selection_epsilon": 0.5}, True),
    ],
)
def test_hdbscan_cluster_selection_matches_sklearn(
    dataframe, queue, cluster_selection_method, params, changes
):
    """scikit-learn's selection on oneDAL's dendrogram must give the same clusters."""
    from sklearnex.preview.cluster import HDBSCAN

    X_df = _convert_to_dataframe(_hierarchy_data(), sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(cluster_selection_method=cluster_selection_method, **params)
    hdbscan.fit(X_df)
    default = HDBSCAN(cluster_selection_method=cluster_selection_method).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    labels = _as_numpy(hdbscan.labels_)
    assert _same_partition(_as_numpy(default.labels_), labels) != changes

    expected_labels, expected_probabilities = _tree_to_labels(hdbscan)
    assert_array_equal(labels, expected_labels)
    assert_array_equal(_as_numpy(hdbscan.probabilities_), expected_probabilities)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "cluster_selection_method,cluster_selection_epsilon",
    [("eom", 1.5), ("eom", 2.0), ("leaf", 0.5), ("leaf", 2.0)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_cluster_selection_epsilon(
    dataframe, queue, cluster_selection_method, cluster_selection_epsilon, dtype
):
    """No two clusters are closer than 'cluster_selection_epsilon'.

    scikit-learn's own selection fails on most values that have an effect (see
    '_tree_to_labels'), so this checks what the parameter guarantees instead.
    """
    from sklearnex.preview.cluster import HDBSCAN

    X_df = _convert_to_dataframe(
        _hierarchy_data(dtype), sycl_queue=queue, target_df=dataframe
    )
    hdbscan = HDBSCAN(
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit(X_df)
    default = HDBSCAN(cluster_selection_method=cluster_selection_method).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    labels, default_labels = _as_numpy(hdbscan.labels_), _as_numpy(default.labels_)
    assert not _same_partition(default_labels, labels)

    # the samples connected below epsilon are never split between clusters
    (component,) = _partitions(hdbscan._single_linkage_tree_, [cluster_selection_epsilon])
    for samples in (component == c for c in np.unique(component)):
        assert len(np.unique(labels[samples & (labels != -1)])) <= 1

    # epsilon only merges clusters, so the clusters without it are not split either
    # and their samples stay in a cluster
    for cluster in np.unique(default_labels[default_labels != -1]):
        merged = labels[default_labels == cluster]
        assert np.all(merged != -1)
        assert len(np.unique(merged)) == 1


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree"])
@pytest.mark.parametrize("min_samples", [1, None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_leaf_size(dataframe, queue, algorithm, min_samples, dtype):
    """'leaf_size' only changes how the neighbors are searched, never the result."""
    from sklearnex.preview.cluster import HDBSCAN

    X_df = _convert_to_dataframe(
        _hierarchy_data(dtype), sycl_queue=queue, target_df=dataframe
    )
    brute = HDBSCAN(algorithm="brute", min_samples=min_samples).fit(X_df)
    results = [
        HDBSCAN(algorithm=algorithm, leaf_size=leaf_size, min_samples=min_samples).fit(
            X_df
        )
        for leaf_size in (1, 5, 40, 1000)
    ]
    reference = results[2]
    for hdbscan in [brute] + results:
        assert hasattr(hdbscan, "_onedal_estimator")

    for hdbscan in results:
        _assert_same_hierarchy(
            reference._single_linkage_tree_, hdbscan._single_linkage_tree_, _RTOL[dtype]
        )
    _assert_same_hierarchy(
        brute._single_linkage_tree_,
        reference._single_linkage_tree_,
        _RTOL_BRUTE[dtype],
    )

    # 'min_samples=1' makes the mutual reachability the plain distance, which has
    # no ties here, so the dendrogram and the clusters drawn from it are unique;
    # with ties only the hierarchy above is
    if min_samples != 1:
        return
    labels = _as_numpy(reference.labels_)
    probabilities = _as_numpy(reference.probabilities_)
    for hdbscan in results:
        _assert_same_partition(labels, hdbscan.labels_)
        assert_allclose(
            _as_numpy(hdbscan.probabilities_), probabilities, rtol=_RTOL[dtype]
        )
    _assert_same_partition(labels, brute.labels_)
    assert_allclose(
        _as_numpy(brute.probabilities_), probabilities, rtol=_RTOL_BRUTE[dtype]
    )


def _assert_tree_structure(tree):
    """A binary dendrogram in scikit-learn's layout, whatever the order of ties."""
    from sklearn.cluster._hdbscan._tree import HIERARCHY_dtype

    n_samples = len(tree) + 1
    assert tree.dtype == HIERARCHY_dtype
    # infinite for the non-finite samples, merged last
    assert np.all(tree["value"][1:] >= tree["value"][:-1])

    # row 'i' creates the node 'n_samples + i' out of two earlier nodes, and every
    # node but the root is merged exactly once
    children = np.concatenate([tree["left_node"], tree["right_node"]])
    rows = np.concatenate([np.arange(len(tree))] * 2)
    assert np.all(children < n_samples + rows)
    assert_array_equal(np.sort(children), np.arange(2 * n_samples - 2))

    sizes = np.concatenate([np.ones(n_samples, dtype=np.int64), tree["cluster_size"]])
    assert_array_equal(
        tree["cluster_size"], sizes[tree["left_node"]] + sizes[tree["right_node"]]
    )
    assert tree["cluster_size"][-1] == n_samples


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize("min_samples", [1, None, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_single_linkage_tree_structure(
    dataframe, queue, algorithm, min_samples, dtype
):
    """'_single_linkage_tree_' must be a well-formed dendrogram of all the samples."""
    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(algorithm=algorithm, min_samples=min_samples).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    tree = hdbscan._single_linkage_tree_
    assert tree.shape == (len(X) - 1,)
    _assert_tree_structure(tree)
    # every merge distance is a mutual reachability distance, so at least the core
    # distance of a sample, which is positive for these distinct samples unless
    # 'min_samples' counts only the sample itself
    assert np.all(np.isfinite(tree["value"]))
    assert np.all(tree["value"] > 0)


# The cosine distance is one minus a dot product, rounded to the absolute precision
# of float32 however small the distance.
_ATOL_COSINE = 1e-6
# the metrics and algorithms oneDAL computes, each with the tolerance of its float32
# merge distances against scikit-learn's float64 ones
_METRIC_CASES = [
    ("euclidean", None, "brute", _RTOL_BRUTE[np.float32]),
    ("euclidean", None, "kd_tree", _RTOL[np.float32]),
    ("euclidean", None, "ball_tree", _RTOL[np.float32]),
    ("manhattan", None, "brute", _RTOL[np.float32]),
    ("manhattan", None, "kd_tree", _RTOL[np.float32]),
    ("manhattan", None, "ball_tree", _RTOL[np.float32]),
    ("chebyshev", None, "brute", _RTOL[np.float32]),
    ("chebyshev", None, "kd_tree", _RTOL[np.float32]),
    ("chebyshev", None, "ball_tree", _RTOL[np.float32]),
    ("minkowski", {"p": 3}, "brute", _RTOL[np.float32]),
    ("minkowski", {"p": 3}, "kd_tree", _RTOL[np.float32]),
    ("minkowski", {"p": 3}, "ball_tree", _RTOL[np.float32]),
    ("cosine", None, "brute", _RTOL[np.float32]),
]


@pytest.mark.parametrize("metric,metric_params,algorithm,rtol32", _METRIC_CASES)
@pytest.mark.parametrize("min_samples", [1, None, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_hierarchy_matches_sklearn(
    metric, metric_params, algorithm, rtol32, min_samples, dtype
):
    """The merge distances and the partitions between them are scikit-learn's.

    Only the order of tied merges, and so the rows of the dendrogram, may differ.
    """
    X = _hierarchy_data(dtype)
    expected, result = _fit_both(
        X,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        min_samples=min_samples,
    )
    assert hasattr(result, "_onedal_estimator")
    single = dtype == np.float32
    _assert_same_hierarchy(
        expected._single_linkage_tree_,
        result._single_linkage_tree_,
        rtol32 if single else _RTOL[np.float64],
        _ATOL_COSINE if single and metric == "cosine" else 0.0,
    )


def _assert_selection_matches(hdbscan, dtype=np.float64):
    """'labels_' and 'probabilities_' are scikit-learn's selection on the tree."""
    labels = _as_numpy(hdbscan.labels_)
    probabilities = _as_numpy(hdbscan.probabilities_)
    expected_labels, expected_probabilities = _tree_to_labels(hdbscan)
    assert_array_equal(labels, expected_labels)
    # oneDAL computes the strengths in the dtype of the data
    if dtype == np.float64:
        assert_array_equal(probabilities, expected_probabilities)
    else:
        assert_allclose(probabilities, expected_probabilities, rtol=1e-6, atol=1e-7)

    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(probabilities[labels == -1] == 0)
    assert np.all(probabilities[labels != -1] > 0)
    for label in np.unique(labels[labels != -1]):
        assert_allclose(probabilities[labels == label].max(), 1.0)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("cluster_selection_method", ["eom", "leaf"])
@pytest.mark.parametrize("allow_single_cluster", [False, True])
@pytest.mark.parametrize("min_cluster_size,min_samples", [(5, None), (5, 1), (15, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_selection_on_own_tree(
    dataframe,
    queue,
    cluster_selection_method,
    allow_single_cluster,
    min_cluster_size,
    min_samples,
    dtype,
):
    """scikit-learn's selection on oneDAL's dendrogram reproduces oneDAL's clusters."""
    from sklearnex.preview.cluster import HDBSCAN

    X_df = _convert_to_dataframe(
        _hierarchy_data(dtype), sycl_queue=queue, target_df=dataframe
    )
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    ).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")
    _assert_selection_matches(hdbscan, dtype)


def test_hdbscan_allow_single_cluster():
    """Data with a single cluster is one cluster only if 'allow_single_cluster'."""
    from sklearnex.preview.cluster import HDBSCAN

    X = np.random.default_rng(0).normal(size=(100, 2))
    labels = {}
    for allow_single_cluster in (False, True):
        hdbscan = HDBSCAN(allow_single_cluster=allow_single_cluster).fit(X)
        assert hasattr(hdbscan, "_onedal_estimator")
        _assert_selection_matches(hdbscan)
        labels[allow_single_cluster] = _as_numpy(hdbscan.labels_)
    assert set(labels[True].tolist()) <= {-1, 0}
    assert not _same_partition(labels[False], labels[True])


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("min_samples", [1, 2, 3, 5])
def test_hdbscan_duplicates(algorithm, min_samples):
    """Repeated samples merge at distance zero, then as scikit-learn merges them."""
    X = np.repeat(_hierarchy_data()[::4], 3, axis=0)
    expected, result = _fit_both(X, algorithm=algorithm, min_samples=min_samples)
    assert hasattr(result, "_onedal_estimator")

    tree = result._single_linkage_tree_
    _assert_tree_structure(tree)
    # brute force expands the squared distance into norms and a dot product, in
    # oneDAL and scikit-learn alike, so copies are only about sqrt(eps * |x|^2) apart
    atol = 1e-6 if algorithm == "brute" else 0.0
    # while 'min_samples' does not exceed the copies of a sample, its core distance
    # is zero, and so are the merges of its copies
    n_zero = np.count_nonzero(tree["value"] <= atol)
    assert n_zero == (2 * len(X) // 3 if min_samples <= 3 else 0)
    _assert_same_hierarchy(expected._single_linkage_tree_, tree, _RTOL[np.float64], atol)
    _assert_selection_matches(result)


@pytest.mark.parametrize(
    "n_samples,min_cluster_size,min_samples",
    [(2, 2, 1), (2, 2, None), (3, 2, 1), (3, 2, None), (3, 3, None), (5, 5, None)],
)
@pytest.mark.parametrize("allow_single_cluster", [False, True])
def test_hdbscan_few_samples(
    n_samples, min_cluster_size, min_samples, allow_single_cluster
):
    """The smallest data oneDAL takes still gives scikit-learn's hierarchy."""
    X = _hierarchy_data()[:: len(_hierarchy_data()) // n_samples][:n_samples]
    expected, result = _fit_both(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=allow_single_cluster,
    )
    assert hasattr(result, "_onedal_estimator")
    tree = result._single_linkage_tree_
    assert tree.shape == (n_samples - 1,)
    _assert_tree_structure(tree)
    _assert_same_hierarchy(expected._single_linkage_tree_, tree, _RTOL[np.float64])
    _assert_selection_matches(result)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_non_finite_rows(dataframe, queue, algorithm, dtype):
    """Non-finite samples are left out, then labelled and merged as scikit-learn does.

    scikit-learn clusters the finite samples alone, labels the samples with an
    infinite value -2 with a strength of 0 and those with a NaN -3 with a NaN
    strength, and merges all of them last, at an infinite distance.
    """
    from sklearn.cluster._hdbscan.hdbscan import remap_single_linkage_tree

    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data(dtype)
    infinite, missing = [7, 50, 130], [3, 99, 200, 201]
    X[infinite[0], 0] = np.inf
    X[infinite[1], 1] = -np.inf
    X[infinite[2]] = np.inf
    X[missing[0], 2] = np.nan
    X[missing[1]] = np.nan
    X[missing[2]] = [np.nan, np.inf, 0]
    # infinities of both signs sum to NaN, which scikit-learn takes as missing
    X[missing[3]] = [np.inf, -np.inf, 0]
    finite = np.setdiff1d(np.arange(len(X)), infinite + missing)

    def fit(data):
        data = _convert_to_dataframe(data, sycl_queue=queue, target_df=dataframe)
        hdbscan = HDBSCAN(algorithm=algorithm).fit(data)
        assert hasattr(hdbscan, "_onedal_estimator")
        return hdbscan

    result, finite_only = fit(X), fit(X[finite])

    labels = _as_numpy(result.labels_)
    probabilities = _as_numpy(result.probabilities_)
    assert_array_equal(labels[infinite], -2)
    assert_array_equal(probabilities[infinite], 0)
    assert_array_equal(labels[missing], -3)
    assert np.all(np.isnan(probabilities[missing]))
    assert_array_equal(labels[finite], _as_numpy(finite_only.labels_))
    assert_array_equal(probabilities[finite], _as_numpy(finite_only.probabilities_))

    expected_tree = remap_single_linkage_tree(
        finite_only._single_linkage_tree_,
        dict(enumerate(finite)),
        non_finite=set(infinite + missing),
    )
    assert_array_equal(result._single_linkage_tree_, expected_tree)
    _assert_tree_structure(result._single_linkage_tree_)
    n_non_finite = len(infinite) + len(missing)
    assert np.all(np.isinf(result._single_linkage_tree_["value"][-n_non_finite:]))


def _components_at(tree, cut_distance, min_cluster_size):
    """The DBSCAN clustering at a cut: components that are large enough, else noise."""
    (component,) = _partitions(tree, [cut_distance])
    sizes = np.bincount(component)
    return np.where(sizes[component] >= min_cluster_size, component, -1)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("min_cluster_size", [2, 5, 30])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_dbscan_clustering_cuts(dataframe, queue, min_cluster_size, dtype):
    """'dbscan_clustering' gives the components of the hierarchy below the cut.

    The cuts are halfway between merge distances, where the components are the
    same whatever the order of tied merges, and so equal scikit-learn's.
    """
    from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN

    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    result = HDBSCAN().fit(X_df)
    assert hasattr(result, "_onedal_estimator")
    expected = _sklearn_HDBSCAN(copy=True).fit(X)

    distances = np.unique(_merge_distances(result))
    tolerance = _RTOL[dtype] * distances[-1]
    for quantile in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        low = distances[int(quantile * (len(distances) - 1))]
        high = distances[distances > low + 4 * tolerance][0]
        cut_distance = (low + high) / 2
        labels = result.dbscan_clustering(cut_distance, min_cluster_size)
        _assert_same_partition(
            _components_at(result._single_linkage_tree_, cut_distance, min_cluster_size),
            labels,
        )
        _assert_same_partition(
            expected.dbscan_clustering(cut_distance, min_cluster_size), labels
        )


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "metric,metric_params", [("euclidean", None), ("manhattan", None), ("cosine", None)]
)
@pytest.mark.parametrize("cluster_selection_method", ["eom", "leaf"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_centers_formulas(
    dataframe, queue, metric, metric_params, cluster_selection_method, dtype
):
    """The centers follow scikit-learn's formulas on oneDAL's own clusters.

    A centroid is the mean of the samples of its cluster weighted by their
    strengths, a medoid the sample that minimizes the strength-weighted sum of the
    distances to the others.
    """
    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    hdbscan = HDBSCAN(
        metric=metric,
        metric_params=metric_params,
        cluster_selection_method=cluster_selection_method,
        store_centers="both",
    ).fit(X_df)
    assert hasattr(hdbscan, "_onedal_estimator")

    labels = _as_numpy(hdbscan.labels_)
    probabilities = _as_numpy(hdbscan.probabilities_)
    centroids, medoids = _as_numpy(hdbscan.centroids_), _as_numpy(hdbscan.medoids_)
    n_clusters = labels.max() + 1
    assert n_clusters > 1
    assert centroids.shape == medoids.shape == (n_clusters, X.shape[1])

    X = X.astype(np.float64)
    for label in range(n_clusters):
        samples, weights = X[labels == label], probabilities[labels == label]
        assert_allclose(
            centroids[label],
            np.average(samples, weights=weights, axis=0),
            rtol=_RTOL[dtype],
            atol=_RTOL[dtype],
        )

        # TODO: also require the smallest strength-weighted cost once
        # uxlfoundation/oneDAL#3883 is in the nightly; until then a fit occasionally
        # returns another member of the cluster
        assert np.all(samples == medoids[label], axis=1).any()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hdbscan_deterministic(dataframe, queue, algorithm, dtype):
    """Two fits on the same data give bitwise the same results."""
    from sklearnex.preview.cluster import HDBSCAN

    X_df = _convert_to_dataframe(
        _hierarchy_data(dtype), sycl_queue=queue, target_df=dataframe
    )
    first, second = (
        HDBSCAN(algorithm=algorithm, store_centers="both").fit(X_df) for _ in range(2)
    )
    for hdbscan in (first, second):
        assert hasattr(hdbscan, "_onedal_estimator")
    assert_array_equal(first._single_linkage_tree_, second._single_linkage_tree_)
    # TODO: add "medoids_" back once oneDAL picks the same medoid on every fit; on
    # the 4-vCPU AMD CI runners a float64 fit occasionally returns another member
    for attribute in ("labels_", "probabilities_", "centroids_"):
        assert_array_equal(
            _as_numpy(getattr(first, attribute)), _as_numpy(getattr(second, attribute))
        )


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
def test_hdbscan_float32_matches_float64(algorithm):
    """float32 data gives the float64 hierarchy, to float32 precision."""
    from sklearnex.preview.cluster import HDBSCAN

    X = _hierarchy_data()
    single, double = (
        HDBSCAN(algorithm=algorithm).fit(X.astype(dtype))
        for dtype in (np.float32, np.float64)
    )
    for hdbscan, dtype in ((single, np.float32), (double, np.float64)):
        assert hasattr(hdbscan, "_onedal_estimator")
        assert hdbscan.probabilities_.dtype == dtype
        assert hdbscan._single_linkage_tree_["value"].dtype == np.float64
    _assert_same_hierarchy(
        double._single_linkage_tree_,
        single._single_linkage_tree_,
        (_RTOL_BRUTE if algorithm == "brute" else _RTOL)[np.float32],
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
    """Labels and the dendrogram are typed as scikit-learn's, the rest as the data."""
    X = _grouped_data(dtype)
    expected, result = _fit_both(
        X, min_cluster_size=_MIN_CLUSTER_SIZE, store_centers="both"
    )
    assert hasattr(result, "_onedal_estimator")
    for attribute in ("labels_", "_single_linkage_tree_"):
        assert getattr(result, attribute).dtype == getattr(expected, attribute).dtype
    for attribute in ("probabilities_", "centroids_", "medoids_"):
        assert getattr(result, attribute).dtype == dtype
    assert result._min_samples == expected._min_samples
    assert result._metric_params == expected._metric_params


if sklearn_check_version("1.9"):
    import array_api_strict
    from sklearn.utils._array_api import (
        get_namespace_and_device,
        move_estimator_to,
        move_to,
    )

    from daal4py.sklearn._utils import _package_check_version
    from onedal.tests.utils._dataframes_support import (
        dpnp_available,
        dpnp_to_strict_working,
        torch_available,
        torch_xpu_available,
    )
    from onedal.tests.utils._device_selection import (
        is_sycl_device_available,
    )
    from sklearnex.tests.utils.misc import assert_same_namespace

    if dpnp_available:
        import dpnp
    if torch_available:
        import torch

    _ARRAYS_LIKE = (
        [np.arange(1), array_api_strict.arange(1)]
        + (
            [dpnp.arange(1, device="gpu")]
            if dpnp_available and is_sycl_device_available("gpu")
            else []
        )
        # Note: 'move_to' has issues with Torch inputs
        # in older sklearn versions.
        + (
            [torch.arange(1, device="xpu")]
            if torch_available and torch_xpu_available and sklearn_check_version("1.10")
            else []
        )
    )

    @pytest.mark.skipif(
        not sklearn_check_version("1.9"),
        reason="Functionality introduced in later sklearn versions",
    )
    @pytest.mark.skipif(
        not _package_check_version("2.2", np.__version__),
        reason="Requires more recent NumPy version",
    )
    @pytest.mark.parametrize("non_finite", [False, True])
    @pytest.mark.parametrize("array_input_like", _ARRAYS_LIKE)
    @pytest.mark.parametrize("array_move_like", _ARRAYS_LIKE)
    def test_hdbscan_move_estimator_to(
        array_input_like, array_move_like, non_finite, with_array_api
    ):
        # TODO: remove this skip once issue in sklearn is fixed:
        # https://github.com/scikit-learn/scikit-learn/issues/35088
        if (
            isinstance(array_move_like, array_api_strict._array_object.Array)
            and dpnp_available
            and isinstance(array_input_like, dpnp.ndarray)
            and not dpnp_to_strict_working
        ):
            pytest.skip()
        from sklearnex.preview.cluster import HDBSCAN

        X = _grouped_data()
        attributes = ["labels_", "probabilities_"]
        if non_finite:
            X[0, 0] = np.inf
            X[1, 1] = np.nan
            store_centers = None
        else:
            # centers of data with non-finite values fall back to scikit-learn
            store_centers = "both"
            attributes += ["centroids_", "medoids_"]

        xp_in, _, device_in = get_namespace_and_device(array_input_like)
        X_in = move_to(X, xp=xp_in, device=device_in)

        model = HDBSCAN(store_centers=store_centers, copy=True).fit(X_in)
        assert hasattr(model, "_onedal_estimator")

        xp_move, _, device_move = get_namespace_and_device(array_move_like)
        model_moved = move_estimator_to(model, xp=xp_move, device=device_move)

        # TODO: update this once scikit-learn introduces a config option
        # to control whether the attributes are always numpy or follow 'X'
        for attr in attributes:
            assert_same_namespace(getattr(model, attr), array_input_like)
            assert_same_namespace(getattr(model_moved, attr), array_move_like)
            assert_allclose(
                _as_numpy(getattr(model_moved, attr)),
                _as_numpy(getattr(model, attr)),
                rtol=1e-6,
            )

        # the dendrogram is a record array, it stays on the host, while the cut
        # follows the namespace of 'labels_'
        assert isinstance(model_moved._single_linkage_tree_, np.ndarray)
        cut = model_moved.dbscan_clustering(0.5)
        assert_same_namespace(cut, array_move_like)
        assert_array_equal(_as_numpy(cut), _as_numpy(model.dbscan_clustering(0.5)))

        attrs_orig = dir(model)
        attrs_moved = dir(model_moved)
        for attr in attrs_orig:
            if attr != "_onedal_estimator":
                assert attr in attrs_moved

    @pytest.mark.skipif(
        not sklearn_check_version("1.9"),
        reason="Functionality introduced in later sklearn versions",
    )
    @pytest.mark.skipif(
        not _package_check_version("2.2", np.__version__),
        reason="Requires more recent NumPy version",
    )
    @pytest.mark.parametrize("array_move_like", _ARRAYS_LIKE)
    @pytest.mark.allow_sklearn_fallback
    def test_hdbscan_move_estimator_to_after_fallback(array_move_like, with_array_api):
        from sklearnex.preview.cluster import HDBSCAN

        X = _grouped_data()
        X[0, 0] = np.inf
        X[1, 1] = np.nan
        model = HDBSCAN(copy=True).fit(_csr_array(X))
        assert not hasattr(model, "_onedal_estimator")

        xp_move, _, device_move = get_namespace_and_device(array_move_like)
        model_moved = move_estimator_to(model, xp=xp_move, device=device_move)

        assert_same_namespace(model_moved.labels_, array_move_like)
        cut = model_moved.dbscan_clustering(0.5)
        assert_same_namespace(cut, array_move_like)
        assert_array_equal(_as_numpy(cut), _as_numpy(model.dbscan_clustering(0.5)))
