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

import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import load_breast_cancer


def test_sklearnex_import_roc_auc():
    from sklearnex.linear_model import LogisticRegression
    from sklearnex.metrics import roc_auc_score

    X, y = load_breast_cancer(return_X_y=True)
    clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    res = roc_auc_score(y, clf.decision_function(X))
    assert_allclose(res, 0.99, atol=1e-2)


def test_sklearnex_import_pairwise_distances():
    from sklearnex.metrics import pairwise_distances

    rng = np.random.RandomState(0)
    x = np.abs(rng.rand(4), dtype=np.float64)
    x = np.vstack([x, x])
    res = pairwise_distances(x, metric="cosine")
    assert_allclose(res, [[0.0, 0.0], [0.0, 0.0]], atol=1e-2)


def test_sklearnex_import_rbf_kernel():
    from sklearnex.metrics.pairwise import rbf_kernel

    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    gamma = 0.5
    res = rbf_kernel(X, gamma=gamma)
    expected_res = np.exp(
        -gamma * np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
    )
    assert_allclose(res, expected_res, atol=1e-6)
