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
from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
from sklearn.datasets import make_blobs

from daal4py.sklearn._utils import daal_check_build_date, daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

pytestmark = pytest.mark.skipif(
    not (daal_check_version((2026, "P", 200)) and daal_check_build_date(20260814)),
    reason="HDBSCAN requires oneDAL >= 2026.2",
)


def _partition(labels):
    """Cluster memberships without the labels themselves.

    oneDAL and scikit-learn find the same clusters, but do not necessarily
    number them in the same way, so the clusterings are compared as the
    partition of the samples that they induce.
    """
    labels = _as_numpy(labels)
    return {frozenset(np.flatnonzero(labels == label)) for label in np.unique(labels)}


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_import_hdbscan(dataframe, queue, dtype):
    """oneDAL must find the same clusters as scikit-learn."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=42)
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    hdbscan = HDBSCAN(min_cluster_size=10).fit(X_df)
    assert "sklearnex" in hdbscan.__module__
    assert hasattr(hdbscan, "_onedal_estimator")

    expected = _sklearn_HDBSCAN(min_cluster_size=10).fit(X)
    assert _partition(hdbscan.labels_) == _partition(expected.labels_)


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
    """Every metric and algorithm offloaded to oneDAL must match scikit-learn."""
    from sklearnex.cluster import HDBSCAN

    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=42)
    params = {
        "min_cluster_size": 10,
        "metric": metric,
        "metric_params": metric_params,
    }

    hdbscan = HDBSCAN(algorithm=algorithm, **params).fit(X)
    assert hasattr(hdbscan, "_onedal_estimator")

    # scikit-learn picks the algorithm itself, the clustering must not depend on it
    expected = _sklearn_HDBSCAN(**params).fit(X)
    assert _partition(hdbscan.labels_) == _partition(expected.labels_)
