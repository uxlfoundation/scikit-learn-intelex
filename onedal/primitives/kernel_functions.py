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

from onedal import _default_backend as backend
from onedal._device_offload import supports_queue
from onedal.common._backend import BackendFunction
from onedal.utils import _sycl_queue_manager as QM

from ..datatypes import from_table, to_table
from ..utils.validation import _check_array


def _check_inputs(X, Y):
    def check_input(data):
        return _check_array(data, dtype=[np.float64, np.float32], force_all_finite=False)

    X = check_input(X)
    Y = X if Y is None else check_input(Y)
    return X, Y


def _compute_kernel(params, submodule, X, Y):
    # get policy for direct backend calls

    queue = QM.get_global_queue()
    X, Y = to_table(X, Y, queue=queue)
    params["fptype"] = X.dtype
    compute_method = BackendFunction(
        submodule.compute, backend, "compute", no_policy=False
    )
    result = compute_method(params, X, Y)
    return from_table(result.values)


@supports_queue
def linear_kernel(X, Y=None, scale=1.0, shift=0.0, queue=None):
    """Compute the linear kernel between X and Y.

    K(x, y) = scale*dot(x, y^T) + shift
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    scale : float, default=1.0
        Multiplication value to scale the inner product.

    shift : float, default=0.0
        Constant offset added to scaled inner product.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Scaled and shifted Gram matrix output.
    """
    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": scale, "shift": shift},
        backend.linear_kernel,
        X,
        Y,
    )


@supports_queue
def rbf_kernel(X, Y=None, gamma=None, queue=None):
    """Compute the rbf (gaussian) kernel between X and Y.

    K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=None
        If None, defaults to 1.0 / n_features.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        The RBF kernel.
    """

    X, Y = _check_inputs(X, Y)

    gamma = 1.0 / X.shape[1] if gamma is None else gamma
    sigma = np.sqrt(0.5 / gamma)

    return _compute_kernel({"method": "dense", "sigma": sigma}, backend.rbf_kernel, X, Y)


@supports_queue
def poly_kernel(X, Y=None, gamma=1.0, coef0=0.0, degree=3, queue=None):
    """Compute the polynomial kernel between X and Y.

    K(x, y) = (gamma*dot(x, y^T) + coef0)**degree
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=1.0
        Multiplication value to scale the inner product.

    coef0 : float, default=0.0
        Constant offset added to scaled inner product.

    degree : int, default=3
        Kernel degree.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        The polynomial kernel.
    """

    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": gamma, "shift": coef0, "degree": degree},
        backend.polynomial_kernel,
        X,
        Y,
    )


@supports_queue
def sigmoid_kernel(X, Y=None, gamma=1.0, coef0=0.0, queue=None):
    """Compute the sigmoid kernel between X and Y.

        :math:`K(x, y) = \\text{tanh}(\\gamma \\mathbf{x}^T \\mathbf{y}) + \\text{coef}_0)`
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=1.0
        Multiplication value to scale the inner product.

    coef0 : float, default=0.0
        Constant offset added to scaled inner product.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Sigmoid kernel between two arrays.
    """

    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": gamma, "shift": coef0}, backend.sigmoid_kernel, X, Y
    )
