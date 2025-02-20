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

# sklearnex IncrementalPCA example for GPU offloading with DPCtl usm ndarray:
#    SKLEARNEX_PREVIEW=YES python ./incremental_pca_dpctl.py

import dpctl
import dpctl.tensor as dpt

# Import estimator via sklearnex's patch mechanism from sklearn
from sklearnex import patch_sklearn, sklearn_is_patched

# IncrementalPCA is currently in preview module, so extra flag is required
patch_sklearn(preview=True)

# Function that can validate current state of patching
sklearn_is_patched()

# Import estimator from the patched sklearn namespace.
from sklearn.decomposition import IncrementalPCA

# Or just directly import estimator from sklearnex namespace.
# from sklearnex.preview.decomposition import IncrementalPCA

# We create GPU SyclQueue and then put data to dpctl tensor using
# the queue. It allows us to do computation on GPU.
queue = dpctl.SyclQueue("gpu")

incpca = IncrementalPCA()

# We do partial_fit for each batch and then print final result.
X_1 = dpt.asarray([[-1, -1], [-2, -1]], sycl_queue=queue)
result = incpca.partial_fit(X_1)

X_2 = dpt.asarray([[-3, -2], [1, 1]], sycl_queue=queue)
result = incpca.partial_fit(X_2)

X_3 = dpt.asarray([[2, 1], [3, 2]], sycl_queue=queue)
result = incpca.partial_fit(X_3)

X = dpt.concat((X_1, X_2, X_3))
transformed_X = incpca.transform(X)

print(f"Principal components:\n{result.components_}")
print(f"Explained variance ratio:\n{result.explained_variance_ratio_}")
print(f"Transformed data:\n{transformed_X}")

# We put the whole data to fit method, it is split automatically and then
# partial_fit is called for each batch.
incpca = IncrementalPCA(batch_size=3)
X = dpt.asarray([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
result = incpca.fit(X)
transformed_X = incpca.transform(X)

print(f"Principal components:\n{result.components_}")
print(f"Explained variance ratio:\n{result.explained_variance_ratio_}")
print(f"Transformed data:\n{transformed_X}")
