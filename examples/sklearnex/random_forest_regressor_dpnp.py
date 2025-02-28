# ==============================================================================
# Copyright 2023 Intel Corporation
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

# sklearnex RF example for GPU offloading with DPNP ndarray:
#    python ./random_forest_regressor_dpnp.py

import dpctl
import dpnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import estimator via sklearnex's patch mechanism from sklearn
from sklearnex import patch_sklearn, sklearn_is_patched

patch_sklearn()

# Function that can validate current state of patching
sklearn_is_patched()

# Import estimator from the patched sklearn namespace.
from sklearn.ensemble import RandomForestRegressor

# Or just directly import estimator from sklearnex namespace.
from sklearnex.ensemble import RandomForestRegressor

# We create GPU SyclQueue and then put data to dpctl tensor using
# the queue. It allows us to do computation on GPU.
queue = dpctl.SyclQueue("gpu")

X, y = make_regression(
    n_samples=1000, n_features=4, n_informative=2, random_state=0, shuffle=False
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dpnp_X_train = dpnp.asarray(X_train, usm_type="device", sycl_queue=queue)
dpnp_y_train = dpnp.asarray(y_train, usm_type="device", sycl_queue=queue)
dpnp_X_test = dpnp.asarray(X_test, usm_type="device", sycl_queue=queue)

rf = RandomForestRegressor(max_depth=2, random_state=0).fit(dpnp_X_train, dpnp_y_train)

pred = rf.predict(dpnp_X_test)

print("Random Forest regression results:")
print("Ground truth (first 5 observations):\n{}".format(y_test[:5]))
print("Regression results (first 5 observations):\n{}".format(pred[:5]))
print("Are predicted results on GPU: {}".format(pred.sycl_device.is_gpu))
