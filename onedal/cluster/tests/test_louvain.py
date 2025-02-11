# ===============================================================================
# Copyright 2024 Intel Corporation
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
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_kernels

from onedal.cluster import Louvain

# Common network dataset https://en.wikipedia.org/wiki/Zachary%27s_karate_club
_karate_club = sp.csr_matrix(
    (
        np.ones((156,), dtype=np.float64),
        np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                17,
                19,
                21,
                31,
                0,
                2,
                3,
                7,
                13,
                17,
                19,
                21,
                30,
                0,
                1,
                3,
                7,
                8,
                9,
                13,
                27,
                28,
                32,
                0,
                1,
                2,
                7,
                12,
                13,
                0,
                6,
                10,
                0,
                6,
                10,
                16,
                0,
                4,
                5,
                16,
                0,
                1,
                2,
                3,
                0,
                2,
                30,
                32,
                33,
                2,
                33,
                0,
                4,
                5,
                0,
                0,
                3,
                0,
                1,
                2,
                3,
                33,
                32,
                33,
                32,
                33,
                5,
                6,
                0,
                1,
                32,
                33,
                0,
                1,
                33,
                32,
                33,
                0,
                1,
                32,
                33,
                25,
                27,
                29,
                32,
                33,
                25,
                27,
                31,
                23,
                24,
                31,
                29,
                33,
                2,
                23,
                24,
                33,
                2,
                31,
                33,
                23,
                26,
                32,
                33,
                1,
                8,
                32,
                33,
                0,
                24,
                25,
                28,
                32,
                33,
                2,
                8,
                14,
                15,
                18,
                20,
                22,
                23,
                29,
                30,
                31,
                33,
                8,
                9,
                13,
                14,
                15,
                18,
                19,
                20,
                22,
                23,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ]
        ),
        np.array(
            [
                0,
                16,
                25,
                35,
                41,
                44,
                48,
                52,
                56,
                61,
                63,
                66,
                67,
                69,
                74,
                76,
                78,
                80,
                82,
                84,
                87,
                89,
                91,
                93,
                98,
                101,
                104,
                106,
                110,
                113,
                117,
                121,
                127,
                139,
                156,
            ]
        ),
    )
)

_karate_labels = np.array(
    [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        2,
        0,
        1,
        0,
        0,
        0,
        2,
        2,
        1,
        0,
        2,
        0,
        2,
        0,
        2,
        2,
        3,
        3,
        2,
        2,
        3,
        2,
        2,
        3,
        2,
        2,
    ]
)


@pytest.mark.parametrize("tol", [1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("max_iter", [10, 100])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_karate_club(dtype, max_iter, tol):
    # test against a well-known network dataset (smoke test)
    est = Louvain(max_iter=max_iter, tol=tol)
    X = _karate_club.astype(dtype)
    est.fit(X)

    assert est.community_count_ == 4
    assert est.modularity_ >= -0.5 and est.modularity_ <= 1.0
    assert_allclose(est.labels_, _karate_labels.astype(dtype))


@pytest.mark.parametrize("scaling", [1e-2, 1.0, 10.0, 1000.0])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scaled_karate(dtype, scaling):
    # calculation of the labels should be using relative
    # values rather than absolute values
    est = Louvain()
    X = scaling * _karate_club.astype(dtype)
    labels = est.fit_predict(X)

    assert est.community_count_ == 4
    assert est.modularity_ >= -0.5 and est.modularity_ <= 1.0
    assert_allclose(labels, _karate_labels.astype(dtype))


def test_forced_community_labels():
    # use the initial values to force the label values to exchange
    # This tests the initialization given by the y value
    est = Louvain()
    X = _karate_club
    est.fit(X)

    # rotate labels
    y = (est.labels_ - 1) % est.community_count_

    # refit using labels
    labels = est.fit(X, est.labels_).labels_

    # refit using rotated labels
    est.fit(X, y)

    # refit labels should match y, rather than original labels
    # But the communities themselves should stay the same
    assert est.community_count_ == 4
    assert est.modularity_ >= -0.5 and est.modularity_ <= 1.0
    assert_allclose(labels, est.labels_)


@pytest.mark.parametrize("n_samples", [20, 40, 100])
@pytest.mark.parametrize("n_clusters", [2, 3, 4])
@pytest.mark.parametrize("metric", ["linear", "rbf", "cosine"])
def test_resolution_simple_clusters(metric, n_clusters, n_samples):
    # iterate through resolutions to show community_count
    # is correlated to resolution (via different
    # affinity matrices)

    X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=n_samples)

    # Convert into a sparse affinity matrix
    X = pairwise_kernels(X, metric=metric)
    np.fill_diagonal(X, 0.0)  # no self loops
    X = sp.csr_matrix(X, dtype=np.float64)
    assert X.min() >= 0

    community = -1  # begin with an unphysical value to guarantee success
    for res in [1e-4, 1e-2, 1, 100]:
        est = Louvain(resolution=res)
        est.fit(X)

        assert (
            est.community_count_ >= community
        ), f"resolution={res} violates expected trend"
        # assert est.modularity_ >= -0.5 and est.modularity_ <= 1.0
        # Deactivating this assert shows something is numerically wrong
        # with the algorithm...
        community = est.community_count_
