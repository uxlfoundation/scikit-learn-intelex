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
from numpy.testing import assert_allclose
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

from onedal.primitives import linear_kernel, poly_kernel, rbf_kernel, sigmoid_kernel
from onedal.tests.utils._device_selection import (
    get_queues,
    pass_if_not_implemented_for_gpu,
)


@pytest.mark.parametrize("queue", get_queues())
def test_dense_self_linear_kernel(queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Linear kernel for the GPU sycl_queue is buggy.")
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)))

    result = linear_kernel(X, queue=queue)
    expected = np.dot(X, np.array(X).T)
    assert_allclose(result, expected, rtol=1e-15)


def _test_dense_small_linear_kernel(queue, scale, shift, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((15, 4)), dtype=dtype)

    result = linear_kernel(X, Y, scale=scale, shift=shift, queue=queue)
    expected = np.dot(X, np.array(Y).T) * scale + shift

    tol = 1e-14 if result.dtype == np.float64 else 1e-6
    assert_allclose(result, expected, rtol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("shift", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_small_linear_kernel(queue, scale, shift, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Linear kernel for the GPU sycl_queue is buggy.")
    _test_dense_small_linear_kernel(queue, scale, shift, dtype)


@pytest.mark.parametrize("queue", get_queues())
def test_dense_self_rbf_kernel(queue):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)))

    result = rbf_kernel(X, queue=queue)
    expected = sklearn_rbf_kernel(X)

    tol = 1e-5 if result.dtype == np.float32 else 1e-14
    assert_allclose(result, expected, rtol=tol)


def _test_dense_small_rbf_kernel(queue, gamma, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((15, 4)), dtype=dtype)

    result = rbf_kernel(X, Y, gamma=gamma, queue=queue)
    expected = sklearn_rbf_kernel(X, Y, gamma)

    tol = 1e-14 if result.dtype == np.float64 else 1e-5
    assert_allclose(result, expected, rtol=tol)


@pytest.mark.parametrize("gamma", [0.1, None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("queue", get_queues())
def test_dense_small_rbf_kernel(queue, gamma, dtype):
    _test_dense_small_rbf_kernel(queue, gamma, dtype)


@pass_if_not_implemented_for_gpu(reason="Polynomial kernel is not implemented for GPU")
@pytest.mark.parametrize("queue", get_queues())
def test_dense_self_poly_kernel(queue):
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((10, 4)))

    degree = 2
    result = poly_kernel(X, degree=degree, queue=queue)
    expected = np.dot(X, np.array(X).T) ** degree

    assert_allclose(result, expected, rtol=1e-14)


def _test_dense_small_poly_kernel(queue, gamma, coef0, degree, dtype):
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(2 * rng.random_sample((15, 4)), dtype=dtype)

    result = poly_kernel(X, Y, gamma=gamma, coef0=coef0, degree=degree, queue=queue)
    expected = (gamma * np.dot(X, np.array(Y).T) + coef0) ** degree

    tol = 1e-14 if result.dtype == np.float64 else 1e-5
    assert_allclose(result, expected, rtol=tol)


@pass_if_not_implemented_for_gpu(reason="Polynomial kernel is not implemented for GPU")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("coef0", [0.0, 1.0])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_small_poly_kernel(queue, gamma, coef0, degree, dtype):
    _test_dense_small_poly_kernel(queue, gamma, coef0, degree, dtype)


@pass_if_not_implemented_for_gpu(reason="Sigmoid kernel is not implemented for GPU")
@pytest.mark.parametrize("queue", get_queues())
def test_dense_self_sigmoid_kernel(queue):
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((15, 4)))

    result = sigmoid_kernel(X, queue=queue)
    expected = np.tanh(np.dot(X, np.array(X).T))

    assert_allclose(result, expected)


def _test_dense_small_sigmoid_kernel(queue, gamma, coef0, dtype):
    rng = np.random.RandomState(0)
    X = np.array(2 * rng.random_sample((10, 4)), dtype=dtype)
    Y = np.array(2 * rng.random_sample((15, 4)), dtype=dtype)

    result = sigmoid_kernel(X, Y, gamma=gamma, coef0=coef0, queue=queue)
    expected = np.tanh(gamma * np.dot(X, np.array(Y).T) + coef0)

    tol = 1e-14 if result.dtype == np.float64 else 1e-6
    assert_allclose(result, expected, rtol=tol)


@pass_if_not_implemented_for_gpu(reason="Sigmoid kernel is not implemented for GPU")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("gamma", [0.1, 1.0, 2.4])
@pytest.mark.parametrize("coef0", [0.0, 1.0, 5.5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_small_sigmoid_kernel(queue, gamma, coef0, dtype):
    _test_dense_small_sigmoid_kernel(queue, gamma, coef0, dtype)
