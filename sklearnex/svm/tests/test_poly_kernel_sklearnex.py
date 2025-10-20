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
from sklearnex.svm.poly_kernel import poly_kernel as skl_poly_kernel
from onedal.primitives import poly_kernel as onedal_poly_kernel

def test_poly_kernel_basic():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])

    # sklearnex interface result
    K_skl = skl_poly_kernel(X, Y, gamma=0.5, coef0=1.0, degree=2)
    # direct oneDAL result
    K_onedal = onedal_poly_kernel(X, Y, gamma=0.5, coef0=1.0, degree=2)

    # check they are close
    assert np.allclose(K_skl, K_onedal)
