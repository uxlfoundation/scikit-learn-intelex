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
import pytest

import daal4py


@pytest.fixture
def restore_num_threads():
    n_threads = daal4py.num_threads()
    yield
    daal4py.daalinit(n_threads)


# https://github.com/uxlfoundation/oneDAL/issues/3820
@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_cross_entropy_loss_is_stable_across_threads(n_threads, restore_num_threads):
    rng = np.random.default_rng(0)
    n_samples, n_features, n_classes = 200_000, 100, 7
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, n_classes, (n_samples, 1)).astype(np.float64)
    beta = 0.1 * rng.standard_normal((n_classes * (n_features + 1), 1))

    loss = daal4py.optimization_solver_cross_entropy_loss(
        nClasses=n_classes,
        numberOfTerms=n_samples,
        interceptFlag=True,
        resultsToCompute="value|gradient",
    )
    loss.setup(X, y, beta)

    def evaluate(n):
        daal4py.daalinit(n)
        res = loss.compute(X, y, beta)
        return res.valueIdx[0, 0], res.gradientIdx.ravel().copy()

    value_ref, grad_ref = evaluate(1)
    for _ in range(10):
        value, grad = evaluate(n_threads)
        np.testing.assert_allclose(value, value_ref, rtol=1e-10)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-10, atol=1e-12)
