# ==============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from sklearnex.svm.poly_kernel import poly_kernel


def test_poly_kernel_basic():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])

    K = poly_kernel(X, Y, gamma=1.0, coef0=0.0, degree=2)

    # expected polynomial kernel manually
    expected = (np.dot(X, Y.T)) ** 2
    assert np.allclose(K, expected), "Polynomial kernel computation is incorrect"


def test_poly_kernel_default_Y():
    X = np.array([[1, 2], [3, 4]])
    K = poly_kernel(X)
    expected = (np.dot(X, X.T)) ** 3  # default degree=3
    assert np.allclose(K, expected), "Polynomial kernel with Y=None is incorrect"
