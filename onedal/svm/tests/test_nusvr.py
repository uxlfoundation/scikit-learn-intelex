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
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn import datasets
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import NuSVR as SklearnNuSVR

from onedal.svm import NuSVR
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)

synth_params = {"n_samples": 500, "n_features": 100, "random_state": 42}


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_diabetes_simple(queue):
    diabetes = datasets.load_diabetes()
    clf = NuSVR(kernel="linear", C=10.0)
    clf.fit(diabetes.data, diabetes.target, queue=queue)
    assert clf.score(diabetes.data, diabetes.target, queue=queue) > 0.02


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_input_format_for_diabetes(queue):
    diabetes = datasets.load_diabetes()

    c_contiguous_numpy = np.asanyarray(diabetes.data, dtype="float", order="C")
    assert c_contiguous_numpy.flags.c_contiguous
    assert not c_contiguous_numpy.flags.f_contiguous
    assert not c_contiguous_numpy.flags.fnc

    clf = NuSVR(kernel="linear", C=10.0)
    clf.fit(c_contiguous_numpy, diabetes.target, queue=queue)
    dual_c_contiguous_numpy = clf.dual_coef_
    res_c_contiguous_numpy = clf.predict(c_contiguous_numpy, queue=queue)

    f_contiguous_numpy = np.asanyarray(diabetes.data, dtype="float", order="F")
    assert not f_contiguous_numpy.flags.c_contiguous
    assert f_contiguous_numpy.flags.f_contiguous
    assert f_contiguous_numpy.flags.fnc

    clf = NuSVR(kernel="linear", C=10.0)
    clf.fit(f_contiguous_numpy, diabetes.target, queue=queue)
    dual_f_contiguous_numpy = clf.dual_coef_
    res_f_contiguous_numpy = clf.predict(f_contiguous_numpy, queue=queue)
    assert_allclose(dual_c_contiguous_numpy, dual_f_contiguous_numpy)
    assert_allclose(res_c_contiguous_numpy, res_f_contiguous_numpy)


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_predict(queue):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    reg = NuSVR(kernel="linear", C=0.1).fit(X, y, queue=queue)

    linear = np.dot(X, reg.support_vectors_.T)
    dec = np.dot(linear, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X, queue=queue).ravel())

    reg = NuSVR(kernel="rbf", gamma=1).fit(X, y, queue=queue)

    rbfs = rbf_kernel(X, reg.support_vectors_, gamma=reg.gamma)
    dec = np.dot(rbfs, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X, queue=queue).ravel())


def _test_diabetes_compare_with_sklearn(queue, kernel):
    diabetes = datasets.load_diabetes()
    clf_onedal = NuSVR(kernel=kernel, nu=0.25, C=10.0)
    clf_onedal.fit(diabetes.data, diabetes.target, queue=queue)
    result = clf_onedal.score(diabetes.data, diabetes.target, queue=queue)

    clf_sklearn = SklearnNuSVR(kernel=kernel, nu=0.25, C=10.0)
    clf_sklearn.fit(diabetes.data, diabetes.target)
    expected = clf_sklearn.score(diabetes.data, diabetes.target)

    assert result > expected - 1e-5
    assert_allclose(clf_sklearn.intercept_, clf_onedal.intercept_, atol=1e-3)
    assert_allclose(
        clf_sklearn.support_vectors_.shape, clf_sklearn.support_vectors_.shape
    )
    assert_allclose(clf_sklearn.dual_coef_, clf_onedal.dual_coef_, atol=1e-2)


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_diabetes_compare_with_sklearn(queue, kernel):
    _test_diabetes_compare_with_sklearn(queue, kernel)


def _test_synth_rbf_compare_with_sklearn(queue, C, nu, gamma):
    x, y = datasets.make_regression(**synth_params)

    clf = NuSVR(kernel="rbf", gamma=gamma, C=C, nu=nu)
    clf.fit(x, y, queue=queue)
    result = clf.score(x, y, queue=queue)

    clf = SklearnNuSVR(kernel="rbf", gamma=gamma, C=C, nu=nu)
    clf.fit(x, y)
    expected = clf.score(x, y)

    assert result > 0.4
    assert abs(result - expected) < 1e-3


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("gamma", ["scale", "auto"])
@pytest.mark.parametrize("C", [100.0, 1000.0])
@pytest.mark.parametrize("nu", [0.25, 0.75])
def test_synth_rbf_compare_with_sklearn(queue, C, nu, gamma):
    _test_synth_rbf_compare_with_sklearn(queue, C, nu, gamma)


def _test_synth_linear_compare_with_sklearn(queue, C, nu):
    x, y = datasets.make_regression(**synth_params)

    clf = NuSVR(kernel="linear", C=C, nu=nu)
    clf.fit(x, y, queue=queue)
    result = clf.score(x, y, queue=queue)

    clf = SklearnNuSVR(kernel="linear", C=C, nu=nu)
    clf.fit(x, y)
    expected = clf.score(x, y)

    # Linear kernel doesn't work well for synthetic regression
    # resulting in low R2 score
    # assert result > 0.5
    assert abs(result - expected) < 1e-3


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("C", [0.001, 0.1])
@pytest.mark.parametrize("nu", [0.25, 0.75])
def test_synth_linear_compare_with_sklearn(queue, C, nu):
    _test_synth_linear_compare_with_sklearn(queue, C, nu)


def _test_synth_poly_compare_with_sklearn(queue, params):
    x, y = datasets.make_regression(**synth_params)

    clf = NuSVR(kernel="poly", **params)
    clf.fit(x, y, queue=queue)
    result = clf.score(x, y, queue=queue)

    clf = SklearnNuSVR(kernel="poly", **params)
    clf.fit(x, y)
    expected = clf.score(x, y)

    assert result > 0.5
    assert abs(result - expected) < 1e-3


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize(
    "params",
    [
        {"degree": 2, "coef0": 0.1, "gamma": "scale", "C": 100, "nu": 0.25},
        {"degree": 3, "coef0": 0.0, "gamma": "scale", "C": 1000, "nu": 0.75},
    ],
)
def test_synth_poly_compare_with_sklearn(queue, params):
    _test_synth_poly_compare_with_sklearn(queue, params)


@pass_if_not_implemented_for_gpu(reason="nusvr is not implemented")
@pytest.mark.parametrize("queue", get_queues())
def test_pickle(queue):
    diabetes = datasets.load_diabetes()

    clf = NuSVR(kernel="rbf", C=10.0)
    clf.fit(diabetes.data, diabetes.target, queue=queue)
    expected = clf.predict(diabetes.data, queue=queue)

    import pickle

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.predict(diabetes.data, queue=queue)
    assert_array_equal(expected, result)
