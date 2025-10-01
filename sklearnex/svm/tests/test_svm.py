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
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes, load_iris, make_classification

from onedal.svm.tests.test_csr_svm import check_svm_model_equal

try:
    from scipy.sparse import csr_array as csr_class
except ImportError:
    from scipy.sparse import csr_matrix as csr_class

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)


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


@pass_if_not_implemented_for_gpu(reason="csr svm is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_iris(queue, kernel):
    from sklearnex import config_context
    from sklearnex.svm import SVC

    if kernel == "rbf":
        pytest.skip("RBF CSR SVM test failing in 2025.0.")
    iris = load_iris()
    rng = np.random.RandomState(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    sparse_iris_data = sp.csr_matrix(iris.data)

    dataset = sparse_iris_data, iris.target, sparse_iris_data

    with config_context(target_offload=queue):
        clf0 = SVC(kernel=kernel)
        clf1 = SVC(kernel=kernel)
        check_svm_model_equal(queue, clf0, clf1, *dataset, decimal=2)


@pass_if_not_implemented_for_gpu(reason="csr svm is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_diabetes(queue, kernel):
    from sklearnex import config_context
    from sklearnex.svm import SVR

    if kernel == "sigmoid":
        pytest.skip("Sparse sigmoid kernel function is buggy.")
    diabetes = load_diabetes()

    sparse_diabetes_data = sp.csr_matrix(diabetes.data)
    dataset = sparse_diabetes_data, diabetes.target, sparse_diabetes_data

    with config_context(target_offload=queue):
        clf0 = SVR(kernel=kernel, C=0.1)
        clf1 = SVR(kernel=kernel, C=0.1)
        check_svm_model_equal(queue, clf0, clf1, *dataset)


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
@pytest.mark.parametrize("estimator", [SVC, NuSVC])
def test_class_weight(queue):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

    clf = estimator(class_weight={0: 0.1})
    clf.fit(X, y, class_count=2, queue=queue)
    assert_array_almost_equal(clf.predict(X, queue=queue).ravel(), [1] * 6)
