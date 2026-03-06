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
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_breast_cancer

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


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


@pytest.mark.allow_sklearn_fallback
@pytest.mark.skipif(
    not sklearn_check_version("1.8"),
    reason="Functionality introduced in later versions of scikit-learn.",
)
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_pairwise_distances_fallback_on_array_api(dataframe, queue):
    from sklearnex import config_context
    from sklearnex.metrics import pairwise_distances

    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(25, 10))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    with config_context(array_api_dispatch=True):
        _ = pairwise_distances(X)
