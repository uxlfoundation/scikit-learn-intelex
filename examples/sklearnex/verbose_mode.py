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

# sklearnex can help you debug your applications by printing messages on it's invocation
# to allow you to see if stock of accelerated version was used.
# By setting sklearnex logger level to "INFO" you would enable this verbose mode
import logging

logging.getLogger("sklearnex").setLevel(logging.INFO)

# Calling scikit-learn patch - this would enable acceleration on all enabled algorithms
from sklearnex import patch_sklearn

patch_sklearn()

# Remaining non modified scikit-learn code
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
v_measure = v_measure_score(labels_true, labels)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Estimated V-measure score: %f" % v_measure)
