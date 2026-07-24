# ===============================================================================
# Copyright 2023 Intel Corporation
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
from numpy.testing import assert_array_equal

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, "P", 200)):
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import davies_bouldin_score

    from onedal.cluster import KMeans, kmeans_plusplus
    from onedal.tests.utils._device_selection import get_queues

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("n_cluster", [2, 5, 11, 128])
    def test_breast_cancer(queue, dtype, n_cluster):
        X, _ = load_breast_cancer(return_X_y=True)
        X = np.asarray(X).astype(dtype=dtype)
        init_data, _ = kmeans_plusplus(X, n_cluster, random_state=777, queue=queue)
        m = KMeans(n_cluster, init=init_data, max_iter=1)
        res = davies_bouldin_score(X, m.fit(X).predict(X))
        thr = 0.45 if n_cluster < 20 else 0.55
        assert res > thr

    from sklearn.neighbors import NearestNeighbors

    def generate_dataset(n_dim, n_cluster, n_points=None, seed=777, dtype=np.float32):
        # We need some reference value of points for each cluster
        n_points = (n_dim * n_cluster) if n_points is None else n_points

        # Creating generator and generating cluster points
        gen = np.random.Generator(np.random.MT19937(seed))
        cs = gen.uniform(low=-1.0, high=+1.0, size=(n_cluster, n_dim))

        # Finding variances for each cluster using 3 sigma criteria
        # It ensures that point is in the Voronoi cell of cluster
        nn = NearestNeighbors(n_neighbors=2)
        d, i = nn.fit(cs).kneighbors(cs)
        assert_array_equal(i[:, 0], np.arange(n_cluster))
        vs = d[:, 1] / 3

        # Generating dataset
        def gen_one(c):
            params = {"loc": cs[c, :], "scale": vs[c], "size": (n_points, n_dim)}
            return gen.normal(**params)

        data = [gen_one(c) for c in range(n_cluster)]
        data = np.concatenate(data, axis=0)
        gen.shuffle(data, axis=0)

        data = data.astype(dtype)

        return (cs, vs, data)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("n_dim", [3, 12, 17])
    @pytest.mark.parametrize("n_cluster", [2, 15, 61])
    def test_generated_dataset(queue, dtype, n_dim, n_cluster):
        seed = 777 * n_dim * n_cluster
        cs, vs, X = generate_dataset(n_dim, n_cluster, seed=seed, dtype=dtype)

        init_data, _ = kmeans_plusplus(X, n_cluster, random_state=seed, queue=queue)
        m = KMeans(n_cluster, init=init_data, max_iter=3, algorithm="lloyd").fit(X)

        rs_centroids = m.cluster_centers_
        nn = NearestNeighbors(n_neighbors=1)
        d, i = nn.fit(rs_centroids).kneighbors(cs)
        # We have applied 2 sigma rule once
        desired_accuracy = int(0.9973 * n_cluster)
        if d.dtype == np.float64:
            desired_accuracy = desired_accuracy - 1
        correctness = d.reshape(-1) <= (vs * 3)
        exp_accuracy = np.count_nonzero(correctness)

        assert desired_accuracy <= exp_accuracy
