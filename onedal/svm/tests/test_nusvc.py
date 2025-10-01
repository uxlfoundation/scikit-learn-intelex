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
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC as SklearnNuSVC

from onedal.svm import NuSVC
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)


def _test_libsvm_parameters(queue, array_constr, dtype):
    X = array_constr([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=dtype)
    y = array_constr([0, 0, 0, 1, 1, 1], dtype=dtype)

    clf = NuSVC(kernel="linear").fit(X, y, class_count=2, queue=queue)
    assert_array_almost_equal(
        clf.dual_coef_, [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]]
    )
    assert_array_equal(clf.support_, [0, 1, 3, 4])
    assert_array_equal(clf.support_vectors_, X[clf.support_])
    assert_array_equal(clf.intercept_, [0.0])
    assert_array_equal(clf.predict(X, queue=queue).ravel(), y)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("array_constr", [np.array])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_libsvm_parameters(queue, array_constr, dtype):
    _test_libsvm_parameters(queue, array_constr, dtype)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_class_weight(queue):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

    clf = NuSVC(class_weight={0: 0.1})
    clf.fit(X, y, class_count=2, queue=queue)
    assert_array_almost_equal(clf.predict(X, queue=queue).ravel(), [1] * 6)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_sample_weight(queue):
    X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = NuSVC(kernel="linear")
    clf.fit(X, y, sample_weight=[1] * 6, class_count=2, queue=queue)
    assert_array_almost_equal(clf.intercept_, [0.0])


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_decision_function(queue):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=np.float32)
    Y = np.array([1, 1, 1, 2, 2, 2], dtype=np.float32)

    clf = NuSVC(kernel="rbf", gamma=1)
    clf.fit(X, Y, class_count=2, queue=queue)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X, queue=queue).ravel())


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_iris(queue):
    iris = datasets.load_iris()
    class_count = len(np.unique(iris.target))
    clf = NuSVC(kernel="linear").fit(
        iris.data, iris.target, class_count=class_count, queue=queue
    )
    assert accuracy_score(iris.target, clf.predict(iris.data, queue=queue)) > 0.9


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_decision_function_shape(queue):
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    class_count = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = NuSVC(kernel="linear").fit(
        X_train, y_train, class_count=class_count, queue=queue
    )
    dec = clf.decision_function(X_train, queue=queue)
    assert dec.shape == (len(X_train), 10)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_pickle(queue):
    iris = datasets.load_iris()
    class_count = len(np.unique(iris.target))
    clf = NuSVC(kernel="linear").fit(
        iris.data, iris.target, class_count=class_count, queue=queue
    )
    expected = clf.decision_function(iris.data, queue=queue)

    import pickle

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.decision_function(iris.data, queue=queue)
    assert_array_equal(expected, result)


def _test_cancer_rbf_compare_with_sklearn(queue, nu, gamma):
    cancer = datasets.load_breast_cancer()
    class_count = len(np.unique(cancer.target))
    if gamma == "auto":
        _gamma = 1.0 / cancer.data.shape[1]
    elif gamma == "scale":
        _gamma = 1.0 / (cancer.data.shape[1] * cancer.data.var())
    else:
        _gamma = gamma

    clf = NuSVC(kernel="rbf", gamma=_gamma, nu=nu)
    clf.fit(cancer.data, cancer.target, class_count=class_count, queue=queue)
    result = accuracy_score(cancer.target, clf.predict(cancer.data, queue=queue))

    clf = SklearnNuSVC(kernel="rbf", gamma=_gamma, nu=nu)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.4
    assert abs(result - expected) < 1e-4


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("gamma", ["scale", "auto"])
@pytest.mark.parametrize("nu", [0.25, 0.5])
def test_cancer_rbf_compare_with_sklearn(queue, nu, gamma):
    _test_cancer_rbf_compare_with_sklearn(queue, nu, gamma)


def _test_cancer_linear_compare_with_sklearn(queue, nu):
    cancer = datasets.load_breast_cancer()
    class_count = len(np.unique(cancer.target))

    clf = NuSVC(kernel="linear", nu=nu)
    clf.fit(cancer.data, cancer.target, class_count=class_count, queue=queue)
    result = accuracy_score(cancer.target, clf.predict(cancer.data, queue=queue))

    clf = SklearnNuSVC(kernel="linear", nu=nu)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.5
    assert abs(result - expected) < 1e-3


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("nu", [0.25, 0.5])
def test_cancer_linear_compare_with_sklearn(queue, nu):
    _test_cancer_linear_compare_with_sklearn(queue, nu)


def _test_cancer_poly_compare_with_sklearn(queue, params):
    cancer = datasets.load_breast_cancer()
    class_count = len(np.unique(cancer.target))
    # gamma="scale"
    _gamma = 1.0 / (cancer.data.shape[1] * cancer.data.var())
    clf = NuSVC(kernel="poly", gamma=_gamma, **params)
    clf.fit(cancer.data, cancer.target, class_count=class_count, queue=queue)
    result = accuracy_score(cancer.target, clf.predict(cancer.data, queue=queue))

    clf = SklearnNuSVC(kernel="poly", gamma=_gamma, **params)
    clf.fit(cancer.data, cancer.target)
    expected = clf.score(cancer.data, cancer.target)

    assert result > 0.5
    assert abs(result - expected) < 1e-4


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize(
    "params",
    [
        {"degree": 2, "coef0": 0.1, "nu": 0.25},
        {"degree": 3, "coef0": 0.0, "nu": 0.5},
    ],
)
def test_cancer_poly_compare_with_sklearn(queue, params):
    _test_cancer_poly_compare_with_sklearn(queue, params)
