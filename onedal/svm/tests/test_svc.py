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

from os import environ

# sklearn requires manual enabling of Scipy array API support
# if `array-api-compat` package is present in environment
# TODO: create generic approach to handle this for all tests
environ["SCIPY_ARRAY_API"] = "1"


import numpy as np
import pytest
import sklearn.utils.estimator_checks
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from onedal.svm import SVC
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)


def _test_libsvm_parameters(queue, array_constr, dtype):
    X = array_constr([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=dtype)
    y = array_constr([1, 1, 1, 2, 2, 2], dtype=dtype)

    clf = SVC(kernel="linear").fit(X, y, queue=queue)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.0])
    assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("array_constr", [np.array])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_libsvm_parameters(queue, array_constr, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_libsvm_parameters(queue, array_constr, dtype)


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
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(class_weight={1: 0.1})
    clf.fit(X, y, queue=queue)
    assert_array_almost_equal(clf.predict(X, queue=queue), [2] * 6)


@pytest.mark.parametrize("queue", get_queues())
def test_sample_weight(queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(kernel="linear")
    clf.fit(X, y, sample_weight=[1] * 6, queue=queue)
    assert_array_almost_equal(clf.intercept_, [0.0])


@pytest.mark.parametrize("queue", get_queues())
def test_decision_function(queue):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=np.float32)
    Y = np.array([1, 1, 1, 2, 2, 2], dtype=np.float32)

    clf = SVC(kernel="rbf", gamma=1, decision_function_shape="ovo")
    clf.fit(X, Y, queue=queue)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X, queue=queue))


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_iris(queue):
    iris = datasets.load_iris()
    clf = SVC(kernel="linear").fit(iris.data, iris.target, queue=queue)
    assert clf.score(iris.data, iris.target, queue=queue) > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_decision_function_shape(queue):
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # check shape of ovo_decition_function=True
    clf = SVC(kernel="linear", decision_function_shape="ovo").fit(
        X_train, y_train, queue=queue
    )
    dec = clf.decision_function(X_train, queue=queue)
    assert dec.shape == (len(X_train), 10)

    with pytest.raises(ValueError, match="must be either 'ovr' or 'ovo'"):
        SVC(decision_function_shape="bad").fit(X_train, y_train, queue=queue)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_pickle(queue):
    iris = datasets.load_iris()
    clf = SVC(kernel="linear").fit(iris.data, iris.target, queue=queue)
    expected = clf.decision_function(iris.data, queue=queue)

    import pickle

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.decision_function(iris.data, queue=queue)
    assert_array_equal(expected, result)


@pass_if_not_implemented_for_gpu(reason="not implemented")
@pytest.mark.parametrize(
    "queue",
    get_queues("cpu")
    + [
        pytest.param(
            get_queues("gpu"),
            marks=pytest.mark.xfail(
                reason="raises Unimplemented error " "with inconsistent error message"
            ),
        )
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svc_sigmoid(queue, dtype):
    X_train = np.array(
        [[-1, 2], [0, 0], [2, -1], [+1, +1], [+1, +2], [+2, +1]], dtype=dtype
    )
    X_test = np.array([[0, 2], [0.5, 0.5], [0.3, 0.1], [2, 0], [-1, -1]], dtype=dtype)
    y_train = np.array([1, 1, 1, 2, 2, 2], dtype=dtype)
    svc = SVC(kernel="sigmoid").fit(X_train, y_train, queue=queue)

    assert_array_equal(svc.dual_coef_, [[-1, -1, -1, 1, 1, 1]])
    assert_array_equal(svc.support_, [0, 1, 2, 3, 4, 5])
    assert_array_equal(svc.predict(X_test, queue=queue), [2, 2, 1, 2, 1])
