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
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import datasets

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_knn_classifier(dataframe, queue):
    X = _convert_to_dataframe([[0], [1], [2], [3]], sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe([0, 0, 1, 1], sycl_queue=queue, target_df=dataframe)
    neigh = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    y_test = _convert_to_dataframe([[1.1]], sycl_queue=queue, target_df=dataframe)
    pred = _as_numpy(neigh.predict(y_test))
    assert "sklearnex" in neigh.__module__
    assert_allclose(pred, [0])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_knn_regression(dataframe, queue):
    X = _convert_to_dataframe([[0], [1], [2], [3]], sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe([0, 0, 1, 1], sycl_queue=queue, target_df=dataframe)
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    y_test = _convert_to_dataframe([[1.5]], sycl_queue=queue, target_df=dataframe)
    pred = _as_numpy(neigh.predict(y_test)).squeeze()
    assert "sklearnex" in neigh.__module__
    assert_allclose(pred, 0.5)


@pytest.mark.parametrize("algorithm", ["auto", "brute"])
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "estimator",
    [LocalOutlierFactor, NearestNeighbors],
)
def test_sklearnex_kneighbors(algorithm, estimator, dataframe, queue):
    X = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    test = _convert_to_dataframe([[0, 0, 1.3]], sycl_queue=queue, target_df=dataframe)
    neigh = estimator(n_neighbors=2, algorithm=algorithm).fit(X)
    result = neigh.kneighbors(test, 2, return_distance=False)
    result = _as_numpy(result)
    assert "sklearnex" in neigh.__module__
    assert_allclose(result, [[2, 0]])
    result = neigh.kneighbors()


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_lof(dataframe, queue):
    X = [[7, 7, 7], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    lof = LocalOutlierFactor(n_neighbors=2)
    result = lof.fit_predict(X)
    result = _as_numpy(result)
    assert hasattr(lof, "_onedal_estimator")
    assert "sklearnex" in lof.__module__
    assert_allclose(result, [-1, 1, 1, 1])


def test_pickle():
    iris = datasets.load_iris()
    clf = KNeighborsClassifier(2).fit(iris.data, iris.target)
    expected = clf.predict(iris.data)
    import pickle

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.predict(iris.data)
    assert_array_equal(expected, result)


def test_pickle_torch_xpu():
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not available")
    if not torch.xpu.is_available():
        pytest.skip("torch XPU device is not available")

    import pickle

    iris = datasets.load_iris()
    X_train = torch.tensor(iris.data, dtype=torch.float32, device="xpu")
    y_train = torch.tensor(iris.target, dtype=torch.float32, device="xpu")

    clf = KNeighborsClassifier(2, algorithm="brute").fit(X_train, y_train)
    predicted = clf.predict(X_train)
    expected = (
        predicted.cpu().numpy() if hasattr(predicted, "cpu") else np.asarray(predicted)
    )

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    predicted2 = clf2.predict(X_train)
    result = (
        predicted2.cpu().numpy() if hasattr(predicted2, "cpu") else np.asarray(predicted2)
    )
    assert_array_equal(expected, result)


@pytest.mark.allow_sklearn_fallback
def test_knn_classifier_single_class():
    """Test KNeighborsClassifier with single-class data (fallback to sklearn).
    oneDAL does not support single-class classification, so this should
    fallback to sklearn's implementation.
    """
    # Create single-class dataset
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 0, 0])  # All same class

    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(X, y)

    # Should predict the only class
    predictions = clf.predict(X)
    assert_array_equal(predictions, y)
    assert_array_equal(clf.classes_, [0])

    # Test with new data
    X_test = np.array([[1.5, 1.5], [2.5, 2.5]])
    predictions_test = clf.predict(X_test)
    assert_array_equal(predictions_test, [0, 0])


def test_no_p_if_metric_is_not_minkowski():
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(25, 3))
    y = rng.standard_normal(size=X.shape[0])
    knn = KNeighborsRegressor(metric="euclidean", p=2).fit(X, y)
    _ = knn.predict(X)
    assert knn.effective_metric_ == "euclidean"
    assert "p" not in knn.effective_metric_params_


@pytest.mark.allow_sklearn_fallback
def test_p_present_if_metric_is_minkowski():
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(25, 3))
    y = rng.standard_normal(size=X.shape[0])
    knn = KNeighborsRegressor(metric="minkowski", p=3).fit(X, y)
    _ = knn.predict(X)
    assert knn.effective_metric_ == "minkowski"
    assert "p" in knn.effective_metric_params_
    assert knn.effective_metric_params_["p"] == 3
