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

# Calling scikit-learn patch - this would enable acceleration on all
# enabled algorithms. This is most straight forward way of patching
from sklearnex import patch_sklearn

patch_sklearn()

# Function that can validate current state of patching
from sklearnex import sklearn_is_patched

sklearn_is_patched()

# Calling scikit-learn unpatch - this would revert patching for all algorithms
from sklearnex import unpatch_sklearn

unpatch_sklearn()

# Direct import of functions in way aligned with scikit-learn
from sklearnex.neighbors import NearestNeighbors

# Patching can be enabled for selected algorithms/estimators only
patch_sklearn(["sklearn.cluster.DBSCAN"])

# Remaining non modified scikit-learn codes
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
