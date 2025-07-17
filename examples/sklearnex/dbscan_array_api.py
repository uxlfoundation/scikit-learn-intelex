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

import array_api_strict

from sklearnex import config_context, patch_sklearn

patch_sklearn()

from sklearn.cluster import DBSCAN

X = array_api_strict.asarray(
    [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]],
    dtype=array_api_strict.float32,
)

# Could be launched without `config_context(array_api_dispatch=True)`. This context
# manager for sklearnex, only guarantee that in case of the fallback to stock
# scikit-learn, fitted attributes to be from the same Array API namespace as
# the training data.
clustering = DBSCAN(eps=3, min_samples=2).fit(X)

print(f"Fitted labels :\n", clustering.labels_)
