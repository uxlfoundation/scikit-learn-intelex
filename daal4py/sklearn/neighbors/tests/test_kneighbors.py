# ==============================================================================
# Copyright 2020 Intel Corporation
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

import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as ScikitKNeighborsClassifier

from daal4py.sklearn.neighbors import KNeighborsClassifier as DaalKNeighborsClassifier

DISTANCES = ["minkowski"]
ALGORITHMS = ["brute", "kd_tree", "auto"]
WEIGHTS = ["uniform", "distance"]
KS = [1, 3, 7, 15, 31]
N_TRIES = 10
ACCURACY_RATIO = 1.0
LOG_LOSS_RATIO = 1.02
ROC_AUC_RATIO = 0.999
IRIS = load_iris()


def _test_determenistic(distance, algorithm, weight, k):
    x_train, x_test, y_train, y_test = train_test_split(
        IRIS.data, IRIS.target, test_size=0.33, random_state=31
    )

    alg_results = []
    for _ in range(N_TRIES):
        # models
        scikit_model = ScikitKNeighborsClassifier(
            n_neighbors=k,
            weights=weight,
            algorithm=algorithm,
            leaf_size=30,
            p=2,
            metric=distance,
        )
        daal_model = DaalKNeighborsClassifier(
            n_neighbors=k,
            weights=weight,
            algorithm=algorithm,
            leaf_size=30,
            p=2,
            metric=distance,
        )
        # training
        scikit_model.fit(x_train, y_train)
        daal_model.fit(x_train, y_train)
        # predict
        scikit_predict = scikit_model.predict(x_test)
        daal_predict = daal_model.predict(x_test)

        distances, indices = daal_model.kneighbors(x_test)
        alg_results.append((distances, indices, daal_predict))
        # accuracy
        scikit_accuracy = accuracy_score(y_test, scikit_predict)
        daal_accuracy = accuracy_score(y_test, daal_predict)
        ratio = daal_accuracy / scikit_accuracy
        reason = "kNN accuracy: scikit_accuracy={},daal_accuracy={}, ratio={}".format(
            scikit_accuracy, daal_accuracy, ratio
        )
        assert ratio >= ACCURACY_RATIO, reason

        # predict proba
        scikit_predict_proba = scikit_model.predict_proba(x_test)
        daal_predict_proba = daal_model.predict_proba(x_test)
        # log loss
        scikit_log_loss = log_loss(y_test, scikit_predict_proba)
        daal_log_loss = log_loss(y_test, daal_predict_proba)
        ratio = daal_log_loss / scikit_log_loss
        reason = "kNN log_loss: scikit_log_loss={},daal_log_loss={}, ratio={}".format(
            scikit_log_loss, daal_log_loss, ratio
        )
        assert ratio <= LOG_LOSS_RATIO, reason

        # ROC AUC
        scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba, multi_class="ovr")
        daal_roc_auc = roc_auc_score(y_test, daal_predict_proba, multi_class="ovr")
        ratio = daal_roc_auc / scikit_roc_auc
        reason = "kNN roc_auc: scikit_roc_auc={}, daal_roc_auc={}, ratio={}".format(
            scikit_roc_auc, daal_roc_auc, ratio
        )
        assert ratio >= ROC_AUC_RATIO, reason

    for i in range(1, N_TRIES):
        for j, res in enumerate(alg_results[i]):
            reason = "Results are different between runs for {}, {}, {}, k={}".format(
                algorithm, weight, distance, k
            )
            assert (res == alg_results[0][j]).mean() == 1, reason


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("k", KS)
def test_determenistic(distance, algorithm, weight, k):
    _test_determenistic(distance, algorithm, weight, k)
