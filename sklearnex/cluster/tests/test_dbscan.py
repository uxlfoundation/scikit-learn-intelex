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

import array_api_strict
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from daal4py.sklearn._utils import (
    _package_check_version,
    daal_check_version,
    sklearn_check_version,
)
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    dpnp_available,
    dpnp_to_strict_working,
    get_dataframes_and_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import (
    is_sycl_device_available,
)

if sklearn_check_version("1.9"):
    from sklearn.datasets import make_blobs
    from sklearn.utils._array_api import (
        get_namespace_and_device,
        move_estimator_to,
        move_to,
    )

    from sklearnex.tests.utils.misc import assert_same_namespace

if dpnp_available:
    import dpnp
if torch_available:
    import torch


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_dbscan(dataframe, queue):
    from sklearnex.cluster import DBSCAN

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
    assert "sklearnex" in dbscan.__module__

    result = _as_numpy(dbscan.labels_)
    expected = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)
    assert_allclose(expected, result)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"), reason="Requires more recent scikit-learn version"
)
@pytest.mark.skipif(
    not _package_check_version("2.2", np.__version__),
    reason="Requires more recent NumPy version",
)
@pytest.mark.parametrize(
    "array_input_like",
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
    ),
)
@pytest.mark.parametrize(
    "array_move_like",
    [np.arange(1), array_api_strict.arange(1)]
    + (
        [dpnp.arange(1, device="gpu")]
        if dpnp_available and is_sycl_device_available("gpu")
        else []
    )
    + (
        [torch.arange(1, device="xpu")]
        if torch_available and torch_xpu_available and sklearn_check_version("1.10")
        else []
    ),
)
def test_dbscan_move_estimator_to(array_input_like, array_move_like, with_array_api):
    # TODO: remove this skip once issue in sklearn is fixed:
    # https://github.com/scikit-learn/scikit-learn/issues/35088
    if (
        isinstance(array_move_like, array_api_strict._array_object.Array)
        and dpnp_available
        and isinstance(array_input_like, dpnp.ndarray)
        and not dpnp_to_strict_working
    ):
        pytest.skip()
    from sklearnex.cluster import DBSCAN

    X, _ = make_blobs(n_samples=20, random_state=123)

    xp_in, _, device_in = get_namespace_and_device(array_input_like)
    X_in = move_to(X, xp=xp_in, device=device_in)

    model = DBSCAN(eps=3, min_samples=2).fit(X_in)

    xp_move, _, device_move = get_namespace_and_device(array_move_like)
    model_moved = move_estimator_to(model, xp=xp_move, device=device_move)

    # TODO: update this once scikit-learn introduces a config option
    # to control whether the attributes are always numpy or follow 'X'
    for attr in ["core_sample_indices_", "components_", "labels_"]:
        assert_same_namespace(getattr(model, attr), array_input_like)
        assert_same_namespace(getattr(model_moved, attr), array_move_like)
        np.testing.assert_almost_equal(
            _as_numpy(getattr(model, attr)),
            _as_numpy(getattr(model_moved, attr)),
        )

    attrs_orig = dir(model)
    attrs_moved = dir(model_moved)
    for attr in attrs_orig:
        assert attr in attrs_moved


def _metric_data():
    """Three tight blobs on the axes, a far outlier and a zero row.

    The blobs are far apart in every supported distance, including cosine, and
    both extra rows are noise in all of them.
    """
    rng = np.random.RandomState(0)
    centers = 10.0 * np.eye(3)
    X = np.vstack(
        [c + 0.3 * rng.standard_normal((30, 3)) for c in centers]
        + [[[5.0, 5.0, 5.0]], np.zeros((1, 3))]
    )
    return X


def _assert_same_clustering(expected, result):
    """Check the core samples and the partition, up to a renaming of clusters."""
    assert_array_equal(
        _as_numpy(result.core_sample_indices_), expected.core_sample_indices_
    )
    labels = _as_numpy(result.labels_)
    assert_array_equal(labels == -1, expected.labels_ == -1)
    pairs = set(zip(expected.labels_.tolist(), labels.tolist()))
    assert len(pairs) == len(set(expected.labels_.tolist())) == len(set(labels.tolist()))


@pytest.mark.skipif(
    not daal_check_version((2026, "P", 300)),
    reason="oneDAL computes only the euclidean distance before 2026.3",
)
@pytest.mark.parametrize(
    "metric,p,eps",
    [
        ("euclidean", None, 1.5),
        ("l2", None, 1.5),
        ("manhattan", None, 2.5),
        ("l1", None, 2.5),
        ("cityblock", None, 2.5),
        ("chebyshev", None, 1.0),
        ("minkowski", 1, 2.5),
        ("minkowski", 1.5, 2.0),
        ("minkowski", 2, 1.5),
        ("minkowski", 3, 1.5),
        ("cosine", None, 0.05),
    ],
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_dbscan_metrics(dataframe, queue, metric, p, eps):
    from sklearn.cluster import DBSCAN as _sklearn_DBSCAN

    from sklearnex.cluster import DBSCAN

    X = _metric_data()
    params = {"eps": eps, "min_samples": 5, "metric": metric, "p": p}
    expected = _sklearn_DBSCAN(**params).fit(X)
    # the data must exercise the noise and more than one cluster
    assert -1 in expected.labels_ and expected.labels_.max() == 2

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    result = DBSCAN(**params).fit(X_df)
    assert hasattr(result, "_onedal_estimator")
    _assert_same_clustering(expected, result)


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize(
    "params",
    [
        {"metric": lambda a, b: np.abs(a - b).sum()},
        {"metric": "minkowski", "p": 0.5},
        {"metric": "minkowski", "p": 2, "metric_params": {"p": 1}},
        {"metric": "manhattan", "algorithm": "kd_tree"},
        {"metric": "sqeuclidean"},
    ],
)
def test_dbscan_metrics_fall_back(params):
    from sklearn.cluster import DBSCAN as _sklearn_DBSCAN

    from sklearnex.cluster import DBSCAN

    X = _metric_data()
    params = {"eps": 2.5, "min_samples": 5, **params}
    expected = _sklearn_DBSCAN(**params).fit(X)
    result = DBSCAN(**params).fit(X)
    assert not hasattr(result, "_onedal_estimator")
    _assert_same_clustering(expected, result)


@pytest.mark.allow_sklearn_fallback
def test_dbscan_precomputed_falls_back():
    from sklearn.cluster import DBSCAN as _sklearn_DBSCAN
    from sklearn.metrics import pairwise_distances

    from sklearnex.cluster import DBSCAN

    D = pairwise_distances(_metric_data(), metric="manhattan")
    params = {"eps": 2.5, "min_samples": 5, "metric": "precomputed"}
    expected = _sklearn_DBSCAN(**params).fit(D)
    result = DBSCAN(**params).fit(D)
    assert not hasattr(result, "_onedal_estimator")
    _assert_same_clustering(expected, result)
