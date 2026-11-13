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

from sklearnex.svm.poly_kernel import poly_kernel


def test_poly_kernel_dense():
    """Test poly_kernel on dense input arrays."""
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])

    result = poly_kernel(X, Y, degree=2, gamma=0.5, coef0=1)
    expected = sklearn_poly_kernel(X, Y, degree=2, gamma=0.5, coef0=1)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_poly_kernel_sparse_fallback():
    """Test that poly_kernel falls back to sklearn implementation for sparse inputs."""
    X = sparse.csr_matrix([[1, 0], [0, 1]])
    Y = sparse.csr_matrix([[0, 1], [1, 0]])

    result = poly_kernel(X, Y, degree=3, gamma=1.0, coef0=1.0)
    expected = sklearn_poly_kernel(X, Y, degree=3, gamma=1.0, coef0=1.0)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
