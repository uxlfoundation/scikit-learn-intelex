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

import pickle

import array_api_strict
import numpy as np
import pandas as pd
import pytest
import scipy
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.base import is_classifier
from sklearn.datasets import load_diabetes, load_iris, make_classification

from onedal.svm.tests.test_csr_svm import check_svm_model_equal
from sklearnex import config_context

try:
    from scipy.sparse import csr_array as csr_class
except ImportError:
    from scipy.sparse import csr_matrix as csr_class

from daal4py.sklearn._utils import _package_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    dpnp_available,
    get_dataframes_and_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import (
    get_queues,
    is_sycl_device_available,
    pass_if_not_implemented_for_gpu,
)

if dpnp_available:
    import dpnp
if torch_available:
    import torch


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_svc(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("SVC fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import SVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.25, 0.25]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_nusvc(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("NuSVC fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import NuSVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]]
    )
    assert_allclose(_as_numpy(svc.support_), [0, 1, 3, 4])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_svr(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("SVR fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import SVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVR(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.1, 0.1]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_nusvr(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("NuSVR fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import NuSVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVR(kernel="linear", nu=0.9).fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-1.0, 0.611111, 1.0, -0.611111]], rtol=1e-3
    )
    assert_allclose(_as_numpy(svc.support_), [1, 2, 3, 5])


@pass_if_not_implemented_for_gpu(reason="csr svm is not implemented")
@pytest.mark.parametrize(
    "queue",
    get_queues("cpu")
    + [
        pytest.param(
            get_queues("gpu"),
            marks=pytest.mark.xfail(
                reason="raises UnknownError for linear and rbf, "
                "Unimplemented error with inconsistent error message "
                "for poly and sigmoid"
            ),
        )
    ],
)
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_binary_dataset(queue, kernel):
    from sklearnex import config_context
    from sklearnex.svm import SVC

    X, y = make_classification(n_samples=80, n_features=20, n_classes=2, random_state=0)
    sparse_X = sp.csr_matrix(X)

    dataset = sparse_X, y, sparse_X
    with config_context(target_offload=queue):
        clf0 = SVC(kernel=kernel)
        clf1 = SVC(kernel=kernel)
        check_svm_model_equal(queue, clf0, clf1, *dataset)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_iris(kernel):
    from sklearnex.svm import SVC

    iris = load_iris()
    rng = np.random.RandomState(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    sparse_iris_data = sp.csr_matrix(iris.data)

    dataset = sparse_iris_data, iris.target, sparse_iris_data

    clf0 = SVC(kernel=kernel, C=0.01)
    clf1 = SVC(kernel=kernel, C=0.01)
    check_svm_model_equal(None, clf0, clf1, *dataset, decimal=2)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_diabetes(kernel):
    from sklearnex.svm import SVR

    if kernel == "sigmoid":
        pytest.skip("Sparse sigmoid kernel function is buggy.")
    diabetes = load_diabetes()

    sparse_diabetes_data = sp.csr_matrix(diabetes.data)
    dataset = sparse_diabetes_data, diabetes.target, sparse_diabetes_data

    clf0 = SVR(kernel=kernel, C=0.1)
    clf1 = SVR(kernel=kernel, C=0.1)
    check_svm_model_equal(None, clf0, clf1, *dataset)


# https://github.com/uxlfoundation/scikit-learn-intelex/issues/1880
def test_works_with_unsorted_indices():
    from sklearnex.svm import SVC

    X = csr_class(
        (
            np.array(
                [0.44943642, 0.6316672, 0.6316672, 0.44943642, 0.6316672, 0.6316672]
            ),
            np.array([1, 4, 3, 1, 2, 0], dtype=np.int32),
            np.array([0, 3, 6], dtype=np.int32),
        ),
        shape=(2, 5),
    )
    y = np.array([1, 0])
    X_test_single = np.array([[1, 0, 0, 0, 0]], dtype=np.float64)
    X_test_multi = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float64)
    model = SVC(probability=True).fit(X, y)
    pred_single = model.predict_proba(X_test_single)
    pred_multi = model.predict_proba(X_test_multi)[0]
    np.testing.assert_array_equal(
        pred_single.reshape(-1),
        pred_multi.reshape(-1),
    )


@pass_if_not_implemented_for_gpu(reason="class weights are not implemented")
@pytest.mark.parametrize(
    "queue",
    get_queues("cpu")
    + [
        pytest.param(
            get_queues("gpu"),
            marks=pytest.mark.xfail(
                reason="class weights are not implemented but the error is not raised"
            ),
        )
    ],
)
def test_class_weight(queue):
    from sklearnex.svm import SVC, NuSVC

    for estimator in [SVC, NuSVC]:
        X = np.array(
            [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=np.float64
        )
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

        clf = estimator(class_weight={0: 0.1})
        with config_context(target_offload=queue):
            clf.fit(X, y)
            assert_array_almost_equal(clf.predict(X).ravel(), [1] * 6)


# TODO: extend this test to cover multi-class scenarios once
# oneDAL multi-class model creation without fitting is fixed,
# and test also different arguments for 'decision_function_shape'.
@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("gamma", ["auto", "scale"])
@pytest.mark.parametrize("n_classes", [0, 2])  # 0 == regression
@pytest.mark.parametrize("nu_class", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_predict_after_fallback(kernel, gamma, n_classes, nu_class, sparse):
    if kernel == "linear" and gamma != "auto":
        pytest.skip()

    from sklearnex import svm

    rng = np.random.default_rng(seed=123)
    if sparse:
        if _package_check_version("1.15", scipy.__version__):
            X = sp.random(100, 4, 0.1, rng=rng, format="csr")
        else:
            X = sp.random(100, 4, 0.1, random_state=456, format="csr")
    else:
        X = rng.standard_normal(size=(100, 4))
    if n_classes == 0:  # regressor
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(n_classes, size=X.shape[0])
    w = rng.standard_gamma(1, size=X.shape[0])
    w[0] = -1.0

    est_name = ("Nu" if nu_class else "") + "SV" + ("C" if n_classes else "R")
    model = getattr(svm, est_name)(kernel=kernel, gamma=gamma)
    model.fit(X, y, w)
    assert not hasattr(model, "_onedal_estimator")

    model_fresh = pickle.loads(pickle.dumps(model))

    pred = model.predict(X)
    assert hasattr(model, "_onedal_estimator")

    pred_fresh = model_fresh.predict(X)
    assert hasattr(model_fresh, "_onedal_estimator")

    # Note: this differs from the above in that it should
    # have create a oneDAL estimator by this point
    model_deser = pickle.loads(pickle.dumps(model))
    pred_deser = model_deser.predict(X)
    assert hasattr(model_deser, "_onedal_estimator")

    np.testing.assert_almost_equal(pred, pred_fresh)
    np.testing.assert_almost_equal(pred, pred_deser)

    # This reuses scikit-learn classes with attributes from
    # sklearnex in order to verify that predictions match.
    from sklearn import svm as skl_svm

    model_skl = getattr(skl_svm, est_name)()
    model_skl.__dict__.update(**model.__dict__)
    # These are internal attributes that scikit-learn uses but sklearnex doesn't.
    # Note that oneDAL models are never supposed to offload predictions to
    # scikit-learn, so they do not need to define them when there's no fallback.
    if n_classes == 2:
        model_skl._dual_coef_ = -model.dual_coef_
        model_skl._intercept_ = -model.intercept_
    else:
        model_skl._dual_coef_ = model.dual_coef_
        model_skl._intercept_ = model.intercept_
    model_skl.support_vectors_ = model.support_vectors_
    model_skl.dual_coef_ = model.dual_coef_
    model_skl.intercept_ = model.intercept_
    assert model_skl.__class__ != model.__class__

    pred_skl = model_skl.predict(X)
    np.testing.assert_allclose(pred, pred_skl)

    if n_classes > 0:
        decision_fn = model.decision_function(X)
        decision_fn_skl = model_skl.decision_function(X)
        np.testing.assert_allclose(decision_fn, decision_fn_skl)


# This one has no oneDAL involvement, but need to verify that it works as expected.
@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("decision_function_shape", ["ovr", "ovo"])
def test_multi_class_prediction_after_fallback(decision_function_shape):
    from sklearnex.svm import SVC

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(150, 4))
    y = rng.integers(3, size=X.shape[0])
    w = rng.standard_gamma(1, size=X.shape[0])
    w[0] = -1.0

    model = SVC(decision_function_shape=decision_function_shape).fit(X, y, w)
    assert not hasattr(model, "_onedal_estimator")

    pred = model.predict(X)
    assert not hasattr(model, "_onedal_estimator")
    assert np.unique(pred).shape[0] == 3

    decision_fn = model.decision_function(X)
    assert not hasattr(model, "_onedal_estimator")
    assert decision_fn.shape[0] == X.shape[0]
    assert decision_fn.shape[1] == 3


@pytest.mark.allow_sklearn_fallback
def test_multiple_calls_to_fit():
    from sklearnex.svm import SVC

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    y = rng.integers(2, size=X.shape[0])
    w = rng.standard_gamma(1, size=X.shape[0])
    w[0] = -1.0

    model = SVC(kernel="linear").fit(X, y).fit(X, y, w)
    decision_fn = model.decision_function(X)
    expected_decision_fn = X @ model.coef_.reshape(-1) + model.intercept_.reshape(-1)

    np.testing.assert_almost_equal(decision_fn, expected_decision_fn)


# These mimic the behavior of sklearn. If their behavior changes,
# these tests will been to be modified too
@pytest.mark.parametrize("estimator", ["SVC", "SVR", "NuSVC", "NuSVR"])
@pytest.mark.parametrize(
    "X_xp, array_api",
    [(np, False), (np, True)]
    + (
        [(array_api_strict, False), (array_api_strict, True)]
        if _package_check_version("2.0", np.__version__)
        else []
    )
    + ([(torch, True)] if torch_available else [])
    + ([(dpnp, True), (dpnp, False)] if dpnp_available else []),
)
def test_error_on_sparse_predict_with_dense_fit(estimator, X_xp, array_api):
    from sklearnex import svm

    rng = np.random.default_rng(seed=123)
    if _package_check_version("1.15", scipy.__version__):
        X_sp = sp.random(100, 10, 0.3, rng=rng, format="csr")
    else:
        X_sp = sp.random(100, 10, 0.3, random_state=rng, format="csr")
    X = X_sp.toarray()
    y = rng.integers(2, size=X.shape[0])

    X = X_xp.from_dlpack(X)
    y = X_xp.from_dlpack(y)

    with config_context(array_api_dispatch=array_api):
        model = getattr(svm, estimator)().fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X_sp)
        if is_classifier(model):
            with pytest.raises(ValueError):
                model.decision_function(X_sp)


@pytest.mark.parametrize("estimator", ["SVC", "SVR", "NuSVC", "NuSVR"])
@pytest.mark.parametrize("array_api", [False, True])
def test_dense_predict_on_sparse_fit_works(estimator, array_api):
    from sklearnex import svm

    rng = np.random.default_rng(seed=123)
    if _package_check_version("1.15", scipy.__version__):
        X_sp = sp.random(100, 10, 0.3, rng=rng, format="csr")
    else:
        X_sp = sp.random(100, 10, 0.3, random_state=rng, format="csr")
    X = X_sp.toarray()
    y = rng.integers(2, size=X.shape[0])

    with config_context(array_api_dispatch=array_api):
        model = getattr(svm, estimator)().fit(X_sp, y)
        pred_dense = model.predict(X)
        pred_sp = model.predict(X_sp)
        np.testing.assert_allclose(pred_dense, pred_sp)
        if is_classifier(model):
            df_dense = model.decision_function(X)
            df_sp = model.decision_function(X_sp)
            np.testing.assert_allclose(df_dense, df_sp)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(
    not _package_check_version("2.1", np.__version__),
    reason="Array API functionality requires more recent version of NumPy.",
)
@pytest.mark.parametrize("X_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("y_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("w_xp", [None, np, pd, array_api_strict])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_classes", [0, 2, 3])  # 0 == regression
@pytest.mark.parametrize("nu_class", [False, True])
def test_svm_mixed_array_namespaces(
    X_xp, y_xp, w_xp, class_weight, n_classes, nu_class, with_array_api
):
    if class_weight is not None and n_classes == 0:
        pytest.skip()
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    if n_classes == 0:  # regressor
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(n_classes, size=X.shape[0])
    if w_xp is None:
        w = None
    else:
        w = rng.standard_gamma(1, size=X.shape[0])

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
    if w_xp is pd:
        w = pd.Series(w)
    elif w_xp is not None:
        w = w_xp.asarray(w)

    from sklearnex import svm

    est_name = ("Nu" if nu_class else "") + "SV" + ("C" if n_classes else "R")
    model = getattr(svm, est_name)()
    if class_weight is not None:
        model.set_params(class_weight=class_weight)

    model.fit(X, y, w)
    pred = model.predict(X)
    _ = model.score(X, y, w)

    if n_classes == 0:
        assert pred.__class__ == (X.__class__ if X_xp is not pd else np.ndarray)
    else:
        if y_xp is pd:
            assert isinstance(model.classes_, np.ndarray)
        else:
            assert model.classes_.__class__ == y.__class__
        decision_scores = model.decision_function(X)
        if X_xp is pd:
            assert isinstance(decision_scores, np.ndarray)
        else:
            assert decision_scores.__class__ == X.__class__

    if n_classes != 0:
        if y_xp is pd:
            y_xp = np
        pred_is_correct = y_xp.astype(y_xp.asarray(pred == y), y_xp.float32)
        assert y_xp.sum(pred_is_correct) >= (0.25 * int(X.shape[0]))


# Note: SVR does not support running on GPU. Hence, this only tries cases
# in which 'X' is on a CPU device. If GPU support gets added in the future,
# it should also test cases where 'X' is on GPU.
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test checks GPU-specific functionality."
)
@pytest.mark.parametrize(
    "X_xp, X_device",
    ([(torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "cpu")] if dpnp_available else []),
)
@pytest.mark.parametrize(
    "y_xp, y_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else [])
    + [(pd, None)],
)
@pytest.mark.parametrize(
    "w_xp, w_device",
    [(None, None)]
    + ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else [])
    + [(pd, None)],
)
@pytest.mark.parametrize(
    "estimator_class",
    [
        "SVR",
        "NuSVR",
    ],
)
def test_svr_mixed_devices(
    X_xp, y_xp, X_device, y_device, w_xp, w_device, estimator_class, with_array_api
):
    # Re-enable this once bug in scikit-learn is solved:
    # https://github.com/scikit-learn/scikit-learn/issues/34046
    if (torch_available and X_xp is torch) and (y_xp is pd or w_xp is pd):
        pytest.skip("Bug in scikit-learn")
    from sklearnex import svm

    model = getattr(svm, estimator_class)()

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    y = rng.standard_normal(size=X.shape[0])
    w = rng.standard_gamma(1, size=X.shape[0]) if w_xp is not None else None

    X = X_xp.asarray(X, device=X_device)
    if y_xp is pd:
        y = pd.Series(y)
    else:
        y = y_xp.asarray(y, device=y_device)
    if w_xp is pd:
        w = pd.Series(w)
    elif w_xp is not None:
        w = w_xp.asarray(w, device=w_device)

    model.fit(X, y, w)
    pred = model.predict(X)
    assert pred.__class__ == X.__class__
    _ = model.score(X, y, w)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
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
def test_svc_mixed_devices(X_xp, X_device, y_xp, y_device, with_array_api):
    from sklearnex.svm import SVC

    model = SVC()

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    y = rng.integers(2, size=X.shape[0])

    X = X_xp.asarray(X, device=X_device)
    if y_xp is pd:
        y = pd.Series(np.array(["a", "b"])[y])
    else:
        y = y_xp.asarray(y, device=y_device)

    model.fit(X, y)
    pred = model.predict(X)
    if y_xp is pd:
        assert isinstance(pred, np.ndarray)
    else:
        assert pred.__class__ == y.__class__
    decision_scores = model.decision_function(X)
    assert decision_scores.__class__ == X.__class__
    _ = model.score(X, y)
