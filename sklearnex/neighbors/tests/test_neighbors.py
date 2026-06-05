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

import warnings

import array_api_strict
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import datasets
from sklearn.base import is_regressor

if hasattr(sp, "csr_array"):
    CSR_CTOR = sp.csr_array
else:
    CSR_CTOR = sp.csr_matrix

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    dpnp_available,
    get_dataframes_and_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import is_sycl_device_available
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)

if dpnp_available:
    import dpnp
if torch_available:
    import torch


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


# This triggers a fallback on the call to 'predict' by passing
# a sparse matrix, which is not supported by oneDAL. If this
# changes, a fallback would need to be triggered in some other way.
@pytest.mark.allow_sklearn_fallback
def test_no_metric_args_warning_on_fallback():
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(25, 3))
    y = rng.standard_normal(size=X.shape[0])

    X_sp = CSR_CTOR(X)

    knn = KNeighborsRegressor(algorithm="brute").fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = knn.predict(X_sp)


# Note: this doesn't check 'kneighbors_graph', because that function
# transfers the data to NumPy internally, so it will not necessarily
# end up erroring out.
@pytest.mark.skipif(
    not sklearn_check_version("1.9"), reason="Functionality introduced in alter versions."
)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_error_on_incompatible_namespaces(weights, with_array_api):
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(25, 3))
    y = rng.standard_normal(size=X.shape[0])
    Xa = array_api_strict.from_dlpack(X)
    ya = array_api_strict.from_dlpack(y)

    knn = KNeighborsRegressor(weights=weights).fit(X, y)

    with pytest.raises(ValueError, match="same namespace"):
        knn.predict(Xa)
    with pytest.raises(ValueError, match="same namespace"):
        knn.kneighbors(Xa)

    knn = KNeighborsRegressor().fit(Xa, ya)
    with pytest.raises(ValueError, match="same namespace"):
        knn.predict(X)
    with pytest.raises(ValueError, match="same namespace"):
        knn.kneighbors(X)


@pytest.mark.skipif(
    not (sklearn_check_version("1.9") and _package_check_version("2.0", np.__version__),
    reason="Functionality introduced in later scikit-learn versions with numpy array API support.",
)
@pytest.mark.parametrize("X_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("y_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("n_classes", [0, 2, 3])  # 0 == regression
def test_mixed_array_namespaces(X_xp, y_xp, weights, n_classes, with_array_api):
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    if n_classes == 0:  # regressor
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(n_classes, size=X.shape[0])

    if X_xp is pd:
        X = pd.DataFrame(X)
    else:
        X = X_xp.asarray(X)
    if y_xp is pd:
        if n_classes != 0:
            y = np.array(["a", "b", "c"])[y]
        y = pd.Series(y)
    else:
        y = y_xp.asarray(y)

    model = (KNeighborsClassifier if n_classes != 0 else KNeighborsRegressor)(
        weights=weights
    )
    model.fit(X, y)
    pred = model.predict(X)
    _ = model.score(X, y)

    _ = model.kneighbors(X)
    _ = model.kneighbors_graph(X)

    if n_classes != 0:
        proba = model.predict_proba(X)
        if X_xp == pd:
            assert isinstance(proba, np.ndarray)
        else:
            assert proba.__class__ == X.__class__

    if n_classes == 0:
        assert pred.__class__ == (X.__class__ if X_xp is not pd else np.ndarray)
    else:
        assert pred.__class__ == (y.__class__ if y_xp is not pd else np.ndarray)

    # Note: this is a quick check to ensure that the result has the same
    # kind of values as the input. There's no particular justification
    # behind requiring 25% classification accuracy.
    if n_classes != 0:
        if y_xp is pd:
            y_xp = np
        pred_is_correct = y_xp.astype(y_xp.asarray(pred == y), y_xp.float32)
        assert y_xp.sum(pred_is_correct) >= (0.25 * int(X.shape[0]))


@pytest.mark.skipif(
    not (sklearn_check_version("1.9") and _package_check_version("2.0", np.__version__),
    reason="Functionality introduced in later scikit-learn versions with numpy array API support.",
)
@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test checks GPU-specific functionality."
)
@pytest.mark.parametrize(
    "X_xp, X_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else []),
)
@pytest.mark.parametrize(
    "y_xp, y_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else [])
    + [(pd, None)],
)
@pytest.mark.parametrize(
    "estimator",
    [
        KNeighborsRegressor(algorithm="brute"),
        KNeighborsClassifier(algorithm="brute"),
    ],
)
def test_knn_mixed_devices(X_xp, y_xp, X_device, y_device, estimator, with_array_api):
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    if is_regressor(estimator):
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(2, size=X.shape[0])

    X = X_xp.asarray(X, device=X_device)
    if y_xp is pd:
        if is_regressor(estimator):
            y = pd.Series(y)
        else:
            y = pd.Series(np.array(["a", "b"])[y])
    else:
        y = y_xp.asarray(y, device=y_device)

    estimator.fit(X, y)
    pred = estimator.predict(X)
    if is_regressor(estimator):
        assert pred.__class__ == X.__class__
    else:
        if y_xp is pd:
            assert isinstance(pred, np.ndarray)
        else:
            assert pred.__class__ == y.__class__
        proba = estimator.predict_proba(X)
        assert proba.__class__ == X.__class__
    _ = estimator.score(X, y)
