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

from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes import from_table, to_table
from ..utils.validation import _check_array


def _check_inputs(X, Y):
    def check_input(data):
        return _check_array(data, dtype=[np.float64, np.float32], force_all_finite=False)

    X = check_input(X)
    Y = X if Y is None else check_input(Y)
    return X, Y


def _compute_kernel(params, submodule, X, Y, queue):
    policy = _get_policy(queue, X, Y)
    X, Y = to_table(X, Y, queue=queue)
    params["fptype"] = X.dtype
    result = submodule.compute(policy, params, X, Y)
    return from_table(result.values)


def linear_kernel(X, Y=None, scale=1.0, shift=0.0, queue=None):
    """
    Compute the linear kernel between X and Y:
        K(x, y) = scale*dot(x, y^T) + shift
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    scale : float, default=1.0
    shift : float, default=0.0

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": scale, "shift": shift},
        _backend.linear_kernel,
        X,
        Y,
        queue,
    )


def rbf_kernel(X, Y=None, gamma=None, queue=None):
    """
    Compute the rbf (gaussian) kernel between X and Y:
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """

    X, Y = _check_inputs(X, Y)

    gamma = 1.0 / X.shape[1] if gamma is None else gamma
    sigma = np.sqrt(0.5 / gamma)

    return _compute_kernel(
        {"method": "dense", "sigma": sigma}, _backend.rbf_kernel, X, Y, queue
    )


def poly_kernel(X, Y=None, gamma=1.0, coef0=0.0, degree=3, queue=None):
    """
    Compute the poly kernel between X and Y:
        K(x, y) = (scale*dot(x, y^T) + shift)**degree
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    scale : float, default=1.0
    shift : float, default=0.0
    degree : float, default=3

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """

    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": gamma, "shift": coef0, "degree": degree},
        _backend.polynomial_kernel,
        X,
        Y,
        queue,
    )


def sigmoid_kernel(X, Y=None, gamma=1.0, coef0=0.0, queue=None):
    """
    Compute the sigmoid kernel between X and Y:
        K(x, y) = tanh(scale*dot(x, y^T) + shift)
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    scale : float, default=1.0
    shift : float, default=0.0

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """

    X, Y = _check_inputs(X, Y)
    return _compute_kernel(
        {"method": "dense", "scale": gamma, "shift": coef0},
        _backend.sigmoid_kernel,
        X,
        Y,
        queue,
    )
