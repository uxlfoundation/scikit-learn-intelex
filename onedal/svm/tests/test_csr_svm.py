# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse as sp
from sklearn import datasets
from sklearn.datasets import make_classification

from onedal.common._mixin import ClassifierMixin
from onedal.svm import SVC, SVR
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)


def check_svm_model_equal(
    queue, dense_svm, sparse_svm, X_train, y_train, X_test, decimal=6
):
    dense_svm.fit(X_train.toarray(), y_train, queue=queue)
    if sp.issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    sparse_svm.fit(X_train, y_train, queue=queue)
    assert sp.issparse(sparse_svm.support_vectors_)
    assert sp.issparse(sparse_svm.dual_coef_)
    assert_array_almost_equal(
        dense_svm.support_vectors_, sparse_svm.support_vectors_.toarray(), decimal
    )
    assert_array_almost_equal(
        dense_svm.dual_coef_, sparse_svm.dual_coef_.toarray(), decimal
    )
    assert_array_almost_equal(dense_svm.support_, sparse_svm.support_)
    assert_array_almost_equal(
        dense_svm.predict(X_test_dense, queue=queue),
        sparse_svm.predict(X_test, queue=queue),
    )

    if isinstance(dense_svm, ClassifierMixin) and isinstance(sparse_svm, ClassifierMixin):
        assert_array_almost_equal(
            dense_svm.decision_function(X_test_dense, queue=queue),
            sparse_svm.decision_function(X_test, queue=queue),
            decimal,
        )


def _test_simple_dataset(queue, kernel):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    sparse_X = sp.lil_matrix(X)
    Y = [1, 1, 1, 2, 2, 2]

    X2 = np.array([[-1, -1], [2, 2], [3, 2]])
    sparse_X2 = sp.dok_matrix(X2)

    dataset = sparse_X, Y, sparse_X2
    clf0 = SVC(kernel=kernel, gamma=1)
    clf1 = SVC(kernel=kernel, gamma=1)
    check_svm_model_equal(queue, clf0, clf1, *dataset)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize(
    "queue",
    get_queues("cpu")
    + [
        pytest.param(
            get_queues("gpu"),
            marks=pytest.mark.xfail(
                reason="raises UnknownError instead of RuntimeError with unimplemented message"
            ),
        )
    ],
)
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_simple_dataset(queue, kernel):
    _test_simple_dataset(queue, kernel)


def _test_binary_dataset(queue, kernel):
    X, y = make_classification(n_samples=80, n_features=20, n_classes=2, random_state=0)
    sparse_X = sp.csr_matrix(X)

    dataset = sparse_X, y, sparse_X
    clf0 = SVC(kernel=kernel)
    clf1 = SVC(kernel=kernel)
    check_svm_model_equal(queue, clf0, clf1, *dataset)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize(
    "queue",
    get_queues("cpu")
    + [
        pytest.param(
            get_queues("gpu"),
            marks=pytest.mark.xfail(
                reason=(
                    "raises UnknownError for linear and rbf, "
                    "Unimplemented error with inconsistent error message "
                    "for poly and sigmoid"
                )
            ),
        )
    ],
)
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_binary_dataset(queue, kernel):
    _test_binary_dataset(queue, kernel)


def _test_iris(queue, kernel):
    iris = datasets.load_iris()
    rng = np.random.RandomState(0)
    perm = rng.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    sparse_iris_data = sp.csr_matrix(iris.data)

    dataset = sparse_iris_data, iris.target, sparse_iris_data

    clf0 = SVC(kernel=kernel)
    clf1 = SVC(kernel=kernel)
    check_svm_model_equal(queue, clf0, clf1, *dataset, decimal=2)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_iris(queue, kernel):
    if kernel == "rbf":
        pytest.skip("RBF CSR SVM test failing in 2025.0.")
    _test_iris(queue, kernel)


def _test_diabetes(queue, kernel):
    diabetes = datasets.load_diabetes()

    sparse_diabetes_data = sp.csr_matrix(diabetes.data)
    dataset = sparse_diabetes_data, diabetes.target, sparse_diabetes_data

    clf0 = SVR(kernel=kernel, C=0.1)
    clf1 = SVR(kernel=kernel, C=0.1)
    check_svm_model_equal(queue, clf0, clf1, *dataset)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_diabetes(queue, kernel):
    if kernel == "sigmoid":
        pytest.skip("Sparse sigmoid kernel function is buggy.")
    _test_diabetes(queue, kernel)


@pass_if_not_implemented_for_gpu(reason="csr svm is not implemented")
@pytest.mark.xfail(reason="Failed test. Need investigate")
@pytest.mark.parametrize("queue", get_queues())
def test_sparse_realdata(queue):
    data = np.array([0.03771744, 0.1003567, 0.01174647, 0.027069])
    indices = np.array([6, 5, 35, 31])
    indptr = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
        ]
    )
    X = sp.csr_matrix((data, indices, indptr))
    y = np.array(
        [
            1.0,
            0.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            0.0,
            1.0,
            2.0,
            2.0,
            0.0,
            2.0,
            0.0,
            3.0,
            0.0,
            3.0,
            0.0,
            1.0,
            1.0,
            3.0,
            2.0,
            3.0,
            2.0,
            0.0,
            3.0,
            1.0,
            0.0,
            2.0,
            1.0,
            2.0,
            0.0,
            1.0,
            0.0,
            2.0,
            3.0,
            1.0,
            3.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            0.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            2.0,
            2.0,
            0.0,
            1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            2.0,
            2.0,
            1.0,
            3.0,
            1.0,
            1.0,
            0.0,
            1.0,
            2.0,
            1.0,
            1.0,
            3.0,
        ]
    )

    clf = SVC(kernel="linear").fit(X.toarray(), y, queue=queue)
    sp_clf = SVC(kernel="linear").fit(X, y, queue=queue)

    assert_array_equal(clf.support_vectors_, sp_clf.support_vectors_.toarray())
    assert_array_equal(clf.dual_coef_, sp_clf.dual_coef_.toarray())
