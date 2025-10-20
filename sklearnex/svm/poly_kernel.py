# ==============================================================================
# Copyright contributors to the oneDAL project
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
from onedal.primitives import poly_kernel as onedal_poly_kernel

def poly_kernel(X, Y=None, gamma=1.0, coef0=0.0, degree=3, queue=None):
    """
    sklearnex interface for the polynomial kernel using oneDAL backend.

    K(x, y) = (gamma * <x, y> + coef0) ** degree
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Input feature array.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        Optional second feature array. If None, Y = X.

    gamma : float, default=1.0
        Scaling factor for the inner product.

    coef0 : float, default=0.0
        Constant term added to scaled inner product.

    degree : int, default=3
        Degree of the polynomial kernel.

    queue : SyclQueue or None, default=None
        Optional SYCL queue for device execution.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Polynomial kernel Gram matrix.
    """
    return onedal_poly_kernel(X, Y=Y, gamma=gamma, coef0=coef0, degree=degree, queue=queue)
