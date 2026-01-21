#===============================================================================
# Copyright 2021 Intel Corporation
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
#===============================================================================

import sklearn
import sklearn.ensemble
import numpy as np
import daal4py
import copy
import pandas as pd

from sklearn.utils import check_array
from sklearn.metrics import accuracy_score

X = pd.read_csv('./data/batch/iris_x_train.csv', header=None)
y = pd.read_csv('./data/batch/iris_y_train.csv', header=None)
weights = pd.read_csv('./data/batch/iris_sample_weights.csv', header=None)

X_test = pd.read_csv('./data/batch/iris_x_test.csv', header=None)
y_test = pd.read_csv('./data/batch/iris_y_test.csv', header=None)

nRows = y.shape[0]
X = np.ascontiguousarray(X, dtype=np.float32)
y = np.ascontiguousarray(y, dtype=np.float32).reshape(y.shape[0])
weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(weights.shape[0])
X_test = np.ascontiguousarray(X_test, dtype=np.float32)
y_test = np.ascontiguousarray(y_test, dtype=np.float32).reshape(y_test.shape[0])

_supported_dtypes_ = [np.float32, np.float64]
X_test = check_array(X_test, dtype=_supported_dtypes_)
X = check_array(X, ensure_2d=True, dtype=_supported_dtypes_)

print("load data done")

sk_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                                 random_state=1347454587,
                                                 max_depth=None,
                                                 max_features=2,
												 n_jobs=1).fit(X, y, sample_weight=weights)

rf_trees = [e.tree_ for e in sk_clf.estimators_]

mb = daal4py.decision_forest_clf_model_builder(nClasses=sk_clf.n_classes_, nTrees=len(rf_trees))
mb.set_n_features(sk_clf.n_features_)

for tr in rf_trees:
    tr_state = tr.__getstate__()
    n_nodes = tr_state['node_count']
    #print("n_nodes:", n_nodes)
    treeId = mb.create_tree(n_nodes)
    v_ar = tr_state['values']
    n_ar = tr_state['nodes']
    parents = [(None, -3, 0)] * n_nodes
    for i in range(n_nodes):
        (left_child, right_child, feature_idx, feature_value, imp_value, nns, wnns) = n_ar[i]
        vi = v_ar[i]
        node_parent, sk_parent, pos = parents[i]
        if node_parent is None:
            kwds = dict()
        else:
            kwds = {'parent_id': node_parent, 'position': pos}
        if left_child > 0 and right_child > 0:
            # split node
            nodeId = mb.add_split(treeId, feature_idx, feature_value, **kwds)
            parents[left_child] = (copy.copy(nodeId), i, 0)
            parents[right_child] = (copy.copy(nodeId), i, 1)
        else:
            # leaf node
            class_idx = np.argmax(vi[0])
            nodeId = mb.add_leaf(treeId, class_idx, **kwds)

daal_model = mb.model()

# **********d4p training**********
daal_engine_ = daal4py.engines_mt2203(seed=1347454587, fptype='float')

d4p_alg_tr = daal4py.decision_forest_classification_training(
        nClasses=sk_clf.n_classes_,
        fptype='float',
        method='hist',
        nTrees=100,
        observationsPerTreeFraction=1,
        featuresPerNode=2,
        maxTreeDepth=0,
        minObservationsInLeafNode=1,
        engine=daal_engine_,
        impurityThreshold=0.,
        varImportance="MDI",
        resultsToCompute="",
        memorySavingMode=False,
        bootstrap=True,
        minObservationsInSplitNode=2,
        minWeightFractionInLeafNode=0.,
        minImpurityDecreaseInSplitNode=0.,
        maxLeafNodes=0,
        maxBins=256,
        minBinSize=1
    )

weight = weights.reshape(-1, 1)
y = y.reshape(-1, 1)
print(y.shape)
print(X.shape)
print(weight.shape)
dfc_trainingResult = d4p_alg_tr.compute(X, y, weight)
real_daal_model = dfc_trainingResult.model

# use the model to predict
d4p_alg_pred = daal4py.decision_forest_classification_prediction(
    nClasses = int(sk_clf.n_classes_),
    fptype = 'float'
)

sk_pred = sk_clf.predict(X_test)

d4p_predictionResult = d4p_alg_pred.compute(X_test, daal_model)
pred = d4p_predictionResult.prediction
d4p_pred = np.take(sk_clf.classes_, pred.ravel().astype(np.int, casting='unsafe'))

d4p_predictionResult2 = d4p_alg_pred.compute(X_test, real_daal_model)
pred2 = d4p_predictionResult2.prediction
d4p_pred2 = np.take(sk_clf.classes_, pred2.ravel().astype(np.int, casting='unsafe'))

#accuracy from model builder
scikit_accuracy = accuracy_score(sk_pred, y_test)
daal4py_accuracy = accuracy_score(d4p_pred, y_test)

#daal accuracy
real_daal4py_accuracy = accuracy_score(d4p_pred2, y_test)

print("sklearn accuracy ", scikit_accuracy)
print("daal4py accuracy (model builder)", daal4py_accuracy)
print("daal4py accuracy (real daal)", real_daal4py_accuracy)
