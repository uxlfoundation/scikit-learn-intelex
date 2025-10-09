# ===============================================================================
# Copyright 2025 Intel Corporation
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


def _compute_distance(params, submodule, X, Y):
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
def correlation_distances(X, Y=None, queue=None):
    """Compute the correlation distances between X and Y.

    D(x, y) = 1 - correlation_coefficient(x, y)
    
    where correlation_coefficient(x, y) = 
    sum((x - mean(x)) * (y - mean(y))) / (std(x) * std(y) * n)
    
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        The correlation distances.
    """

    X, Y = _check_inputs(X, Y)
    return _compute_distance(
        {"method": "dense"}, backend.correlation_distance, X, Y
    )


@supports_queue
def cosine_distances(X, Y=None, queue=None):
    """Compute the cosine distances between X and Y.

    D(x, y) = 1 - (x · y) / (||x|| * ||y||)
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features)
        An optional second feature array. If `None`, uses `Y=X`.

    queue : SyclQueue or None, default=None
        SYCL Queue object for device code execution. Default
        value None causes computation on host.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        The cosine distances.
    """
    X, Y = _check_inputs(X, Y)
    return _compute_distance(
        {"method": "dense"}, backend.cosine_distance, X, Y
    )
