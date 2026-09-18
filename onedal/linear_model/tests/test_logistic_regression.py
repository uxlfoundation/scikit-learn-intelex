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
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from onedal.linear_model import LogisticRegression
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues("gpu"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_breast_cancer(queue, dtype):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X.astype(dtype), y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    model = LogisticRegression(fit_intercept=True, solver="newton-cg")
    model.fit(X_train, y_train, queue=queue)
    y_pred = model.predict(X_test, queue=queue)
    assert accuracy_score(y_test, y_pred) > 0.95

    assert hasattr(model, "n_iter_")
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "_n_inner_iter")


@pytest.mark.parametrize("queue", get_queues("gpu"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pickle(queue, dtype):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X.astype(dtype), y.astype(dtype)
    model = LogisticRegression(fit_intercept=True, solver="newton-cg")
    model.fit(X, y, queue=queue)
    expected = model.predict(X, queue=queue)

    import pickle

    dump = pickle.dumps(model)
    model2 = pickle.loads(dump)

    assert isinstance(model2, model.__class__)
    result = model2.predict(X, queue=queue)

    assert_array_equal(expected, result)


@pytest.mark.parametrize("queue", get_queues("gpu"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [(3007, 17, 0.05), (50000, 100, 0.01), (512, 10, 0.5)])
def test_csr(queue, dtype, dims):
    n, p, density = dims
    X, y = make_classification(n, p, random_state=42)
    np.random.seed(2007 + n + p)
    mask = np.random.binomial(1, density, (n, p))
    X = X * mask
    X_sp = csr_matrix(X)
    model = LogisticRegression(fit_intercept=True, solver="newton-cg")
    model.fit(X, y, queue=queue)
    pred = model.predict(X, queue=queue)

    model_sp = LogisticRegression(fit_intercept=True, solver="newton-cg")
    model_sp.fit(X_sp, y, queue=queue)
    pred_sp = model_sp.predict(X_sp, queue=queue)

    rtol = 2e-4
    assert_allclose(pred, pred_sp, rtol=rtol)
    assert_allclose(model.coef_, model_sp.coef_, rtol=rtol)
    assert_allclose(model.intercept_, model_sp.intercept_, rtol=rtol)
