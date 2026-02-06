# ==============================================================================
# Copyright 2024 Intel Corporation
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

import dpctl
import dpnp

from sklearnex.linear_model import IncrementalLinearRegression

# We create GPU SyclQueue and then put data to dpnp arrays using
# the queue. It allows us to do computation on GPU.

queue = dpctl.SyclQueue("gpu")

inclin = IncrementalLinearRegression()

# We do partial_fit for each batch and then print final result.
X_1, y_1 = dpnp.asarray([[0, 1], [1, 2]], sycl_queue=queue), dpnp.asarray(
    [2, 4], sycl_queue=queue
)
result = inclin.partial_fit(X_1, y_1)

X_2, y_2 = dpnp.asarray([[2, 3]], sycl_queue=queue), dpnp.asarray([6], sycl_queue=queue)
result = inclin.partial_fit(X_2, y_2)

X_3, y_3 = dpnp.asarray([[0, 2], [1, 3], [2, 4]], sycl_queue=queue), dpnp.asarray(
    [3, 5, 7], sycl_queue=queue
)
result = inclin.partial_fit(X_3, y_3)

print(f"Coefs:\n{result.coef_}")
print(f"Intercept:\n{result.intercept_}")

# We put the whole data to fit method, it is split automatically and then
# partial_fit is called for each batch.
inclin = IncrementalLinearRegression(batch_size=3)
X, y = dpnp.asarray(
    [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [2, 4]], sycl_queue=queue
), dpnp.asarray([2, 4, 6, 3, 5, 7], sycl_queue=queue)
result = inclin.fit(X, y)

print(f"Coefs:\n{result.coef_}")
print(f"Intercept:\n{result.intercept_}")
