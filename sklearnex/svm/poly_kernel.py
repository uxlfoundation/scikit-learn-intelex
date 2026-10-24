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
from scipy import sparse
from sklearn.metrics.pairwise import polynomial_kernel as sklearn_poly_kernel

from onedal.primitives import poly_kernel as onedal_poly_kernel


def poly_kernel(
    X,
    Y=None,
    degree=3,
    gamma=None,
    coef0=1,
    queue=None,
):
    """
    Compute the polynomial kernel using the oneDAL backend when possible.

    Falls back to scikit-learn's ``polynomial_kernel`` for unsupported cases
    such as sparse inputs.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Input feature array.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        Optional second feature array. If None, ``Y = X``.

    degree : int, default=3
        Degree of the polynomial kernel.

    gamma : float, default=None
        Scaling factor for the inner product. If None, ``1 / n_features`` is used.

    coef0 : float, default=1
        Constant term added to the scaled inner product.

    queue : SyclQueue or None, default=None
        Optional SYCL queue for device execution.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Computed polynomial kernel Gram matrix.
    """
    # Fall back to sklearn if sparse input
    if sparse.issparse(X) or (Y is not None and sparse.issparse(Y)):
        return sklearn_poly_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    # Handle gamma default like sklearn
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Use oneDAL accelerated path
    return onedal_poly_kernel(
        X, Y=Y, gamma=gamma, coef0=coef0, degree=degree, queue=queue
    )
